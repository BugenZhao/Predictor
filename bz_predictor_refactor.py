import ast
import io
import time

from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import h5py
import os
import cv2
import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
import tkinter.simpledialog
import tkinter.messagebox
from multiprocessing import Process
from threading import Thread


class PredictorView(tkinter.Tk):
    def __init__(self, controller):
        super().__init__()

        self.controller = controller
        self.model = ''

        # self.maxsize(width=400, height=400)
        self.geometry('300x200')
        self.resizable(False, False)
        self.title('BZ Predictor')

        self.panel = tk.Label(self, text='Preview', height=8)
        self.panel.pack()

        self.button = ttk.Button(self, text='Predict',
                                 command=self.go)
        self.button.pack()

        self.statusbar = ttk.Label(self, text='', anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.update_statusbar()

        self.create_menu()

    def update_title(self):
        self.title('BZ Predictor - ' + self.controller.get_model())

    def update_statusbar(self, text='', final=False):
        if final:
            time.sleep(3)
            self.statusbar.config(text='Ready')
        elif text == '':
            self.statusbar.config(text='Current: ' + self.controller.get_model())
            # Thread(target=self.update_statusbar, args=('', True)).start()
        else:
            self.statusbar.config(text=text)
        self.update()

    def create_menu(self):
        menus = ['Predictor', 'Help']
        items = [['Set model...', '-', 'Quit'], ['About']]
        callbacks = [[self.set_model, None, self.quit], [self.about]]
        menubar = tk.Menu(self)
        for i, x in enumerate(menus):
            m = tk.Menu(menubar, tearoff=0)
            for item, callback in zip(items[i], callbacks[i]):
                if isinstance(item, list):
                    sm = tk.Menu(menubar, tearoff=0)
                    for subitem, subcallback in zip(item, callback):
                        if subitem == '-':
                            sm.add_separator()
                        else:
                            sm.add_command(
                                label=subitem, command=subcallback,
                                compound='left')
                    m.add_cascade(label=item[0], menu=sm)
                elif item == '-':
                    m.add_separator()
                else:
                    m.add_command(label=item, command=callback,
                                  compound='left')
            menubar.add_cascade(label=x, menu=m)
        self.config(menu=menubar)

    def set_model(self):
        new_model = tk.simpledialog.askstring(
            title='Set model...', prompt='New model: ')
        if (new_model is not None) and (new_model != self.model):
            self.model = new_model
            Thread(target=self.controller.update_model, args=(new_model,)).start()

    def go(self):
        cv2.destroyAllWindows()
        path = tk.filedialog.askopenfilename()
        data = {
            0: {
                'bz': path
            }
        }
        self.controller.process(data)

    def about(self):
        tk_version = self.tk.call('info', 'patchlevel')
        tk.messagebox.showinfo(message='BZ Predictor in MVC pattern.' + "\n\nTK version: " + tk_version)


class PredictorUIModel:
    def __init__(self, controller):
        self.prediction = None
        self.controller = controller

    def set_model(self, model) -> bool:
        try:
            new_prediction = BZPrediction(model)
        except:
            return False
        self.prediction = new_prediction
        return True

    def predict(self, data):
        predict_result = self.prediction.predict(data)
        print(predict_result)
        return predict_result


class PredictorController:
    class ButtonLock:
        def __init__(self, view):
            self.view = view
            self.view.button.config(state='disabled')
            self.view.update()

        def __del__(self):
            self.view.button.config(state='normal')
            self.view.update()

    def __init__(self):
        self.model = ''
        self.view = PredictorView(self)
        self.mvc_model = PredictorUIModel(self)

    def run(self, model=None):
        thread = Thread(target=self.update_model, args=(model,))
        thread.start()
        self.view.mainloop()

    def get_model(self):
        return self.model

    def update_model(self, new_model):
        if new_model is None: return
        self.view.update_statusbar(text='Refreshing...')
        lock = self.ButtonLock(self.view)
        flag = self.mvc_model.set_model(new_model)
        if flag:
            self.model = new_model
        else:
            tk.messagebox.showerror(
                title='Oops!', message='There\' s no model named "'
                                       + new_model + '".')
        self.view.update_statusbar()
        self.view.update_title()

    def process(self, data):
        self.view.update_statusbar(text='Processing...')
        lock = self.ButtonLock(self.view)
        try:
            predict_result = self.mvc_model.predict(data)
            self.view.update_statusbar(
                text='Done: ' + str(predict_result['detection_classes']) + ' ' +
                     str(predict_result['detection_scores']))

            self.cv2_show(data, predict_result)

        except Exception as e:
            print(e)
            self.view.update_statusbar('Latest operation was failed or aborted')

        # self.view.update_statusbar()

    def cv2_show(self, data, predict_result):
        img = cv2.imread(data[0]['bz'])
        for clazz, box, score in zip(predict_result['detection_classes'], predict_result['detection_boxes'],
                                     predict_result['detection_scores']):
            x0, y0, x1, y1 = tuple(box)
            cv2.rectangle(img, (y0, x0), (y1, x1),
                          (0, 0, 255), thickness=2)
            cv2.putText(img, str(clazz) + ':' + str(score), (y1, x1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255),
                        thickness=1)
        # cv2.namedWindow('Result', flags=cv2.WINDOW_AUTOSIZE)
        # self.view.canvas.create_image(image=Image.fromarray(img))
        # cv2.imshow('Result', img)
        # cv2.waitKey(0)

        # self.view.panel.config(image=ImageTk.PhotoImage(image=Image.fromarray(img)))

        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
        current_image = Image.fromarray(cv2image)  # convert image for PIL
        imgtk = ImageTk.PhotoImage(image=current_image)  # convert image for tkinter
        self.view.geometry(str(imgtk.width()) + 'x' + str(imgtk.height() + 50))

        self.view.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
        self.view.panel.config(height=0, image=imgtk)  # show the image


class BZPrediction:
    def __init__(self, model):
        self.model_path = model + '/'
        self.predict_fn = tf.contrib.predictor.from_saved_model(
            self.model_path, signature_def_key='predict_object')
        pass

    def pre_process(self, data: dict) -> dict:
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                image = Image.open(file_content)
                image = image.convert('RGB')
                image = np.asarray(image, dtype=np.float32)
                image = image[np.newaxis, :, :, :]
                preprocessed_data[k] = image
        return preprocessed_data

    def do_predict(self, data: dict) -> dict:
        output = self.predict_fn({'images': data[0]})
        return output

    def post_process(self, data: dict) -> dict:
        prob_threshold = 0.3
        nms_iou_threshold = 0.3

        h5f = h5py.File(os.path.join(self.model_path, 'index'), 'r')
        labels_list = h5f['labels_list'][:]
        h5f.close()
        num_boxes = len(data['detection_classes'])
        classes = []
        boxes = []
        scores = []
        result_return = dict()
        for i in range(num_boxes):
            if data['detection_scores'][i] > prob_threshold:
                class_id = data['detection_classes'][i] - 1
                classes.append(labels_list[int(class_id)])
                boxes.append(data['detection_boxes'][i])
                scores.append(data['detection_scores'][i])
        ##########add NMS#######################################
        bounding_boxes = boxes
        confidence_score = scores
        # Bounding boxes
        boxes = np.array(bounding_boxes)
        picked_boxes = []
        picked_score = []
        picked_classes = []
        if len(boxes) != 0:
            # coordinates of bounding boxes
            start_x = boxes[:, 0]
            start_y = boxes[:, 1]
            end_x = boxes[:, 2]
            end_y = boxes[:, 3]
            # Confidence scores of bounding boxes
            score = np.array(confidence_score)
            # Picked bounding boxes
            # Compute areas of bounding boxes
            areas = (end_x - start_x + 1) * (end_y - start_y + 1)
            # Sort by confidence score of bounding boxes
            order = np.argsort(score)
            # Iterate bounding boxes
            while order.size > 0:
                # The index of largest confidence score
                index = order[-1]
                # Pick the bounding box with largest confidence score
                picked_boxes.append(bounding_boxes[index])
                picked_score.append(confidence_score[index])
                picked_classes.append(classes[index])
                # Compute ordinates of intersection-over-union(IOU)
                x1 = np.maximum(start_x[index], start_x[order[:-1]])
                x2 = np.minimum(end_x[index], end_x[order[:-1]])
                y1 = np.maximum(start_y[index], start_y[order[:-1]])
                y2 = np.minimum(end_y[index], end_y[order[:-1]])
                # Compute areas of intersection-over-union
                w = np.maximum(0.0, x2 - x1 + 1)
                h = np.maximum(0.0, y2 - y1 + 1)
                intersection = w * h
                # Compute the ratio between intersection and union
                ratio = intersection / \
                        (areas[index] + areas[order[:-1]] - intersection)
                left = np.where(ratio < nms_iou_threshold)
                order = order[left]
        result_return['detection_classes'] = picked_classes
        result_return['detection_boxes'] = picked_boxes
        result_return['detection_scores'] = picked_score
        return result_return

    def predict(self, data: dict) -> dict:
        a = self.pre_process(data)
        b = self.do_predict(a)
        return self.post_process(b)


if __name__ == '__main__':
    app = PredictorController()
    app.run('park')
