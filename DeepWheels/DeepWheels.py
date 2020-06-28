from DataWriter import DataWriter
import Predict


class DeepWheels:
    def __init__(self):
        self.predict_process = Process(target=Predict.predict, args=('bob',))
        self.writer_process = Process(target=f, args=('bob',))
        self.data_writer = DataWriter()
    
    def predict(self):
        '''
    
        :return:
        '''

