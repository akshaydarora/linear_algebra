import pickle
import os

class SaveLoadLS(object):

    def __init__(self,file_obj=None,file_dir=None,file_name=None):
      self.file_obj=file_obj
      self.file_dir=file_dir
      self.file_name=file_name
      
    def save_file(self):

        if not os.path.exists(self.file_dir):
            os.makedirs(self.file_dir)
        f = open(os.path.join(self.file_dir,'{}.pickle'.format(self.file_name)), 'wb')
        pickle.dump(self.file_obj, f, -1)
        f.close()

    def load_file(self):

        f = open(os.path.join(self.file_dir,'{}.pickle'.format(self.file_name)), 'rb')
        file_obj = pickle.load(f)
        f.close()
        return file_obj