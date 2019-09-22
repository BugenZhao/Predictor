import ast
import io

from PIL import Image
import tensorflow as tf
import numpy as np
import h5py
import os
import cv2
import tkinter
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

        self.maxsize(width=400, height=200)
        self.geometry('400x200')
        self.resizable(False, False)
        self.title('BZ Predictor')

        self.button = tkinter.Button(self, text='Predict', height=1, width=6,
                                     command=self.go)
        self.button.place(relx=0.50, rely=0.50, anchor='center')

        self.statusbar = tkinter.Label(
            self, text='', bd=1, relief=tkinter.SUNKEN, anchor=tkinter.W)
        self.statusbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
        self.update_statusbar()

        self.create_menu()

    def update_statusbar(self, text=''):
        if text == '':
            self.statusbar.config(text='Current: ' + self.controller.get_model())
        else:
            self.statusbar.config(text=text)
        self.update()

    def create_menu(self):
        menus = ['Predictor']
        items = [['Set model...', '-', 'Exit']]
        callbacks = [[self.set_model, None, self.quit]]
        menubar = tkinter.Menu(self)
        for i, x in enumerate(menus):
            m = tkinter.Menu(menubar, tearoff=0)
            for item, callback in zip(items[i], callbacks[i]):
                if isinstance(item, list):
                    sm = tkinter.Menu(menubar, tearoff=0)
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
        new_model = tkinter.simpledialog.askstring(
            title='Set model...', prompt='New model: ')
        if new_model is not None and new_model != self.model:
            self.controller.update_model(new_model)

    def go(self):
        cv2.destroyAllWindows()
        path = tkinter.filedialog.askopenfilename()
        data = {
            0: {
                'bz': path
            }
        }
        self.controller.process(data)


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
        print('asdfasdf')
        if new_model is None: return
        self.view.update_statusbar(text='Refreshing...')
        lock = self.ButtonLock(self.view)
        flag = self.mvc_model.set_model(new_model)
        if flag:
            self.model = new_model
        else:
            tkinter.messagebox.showinfo(
                title='Oops!', message='There\' s no model named "'
                                       + new_model + '".')
        self.view.update_statusbar()

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
            self.view.update_statusbar('Last operation was failed or aborted')

        self.view.update_statusbar()

    def cv2_show(self, data, predict_result):
        img = cv2.imread(data[0]['bz'])
        for clazz, box, score in zip(predict_result['detection_classes'], predict_result['detection_boxes'],
                                     predict_result['detection_scores']):
            x0, y0, x1, y1 = tuple(box)
            cv2.rectangle(img, (y0, x0), (y1, x1),
                          (0, 0, 255), thickness=2)
            cv2.putText(img, str(clazz) + ':' + str(score), (y1, x1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255),
                        thickness=1)
        cv2.namedWindow('Result', flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Result', img)
        cv2.waitKey(0)


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
