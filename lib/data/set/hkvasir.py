from .base import *

class HKVASIR(BaseDataset):
    nb_train_all = 8567
    nb_test_all = 2095
    def __init__(self, root, classes, dl_type='train', transform=None):
        BaseDataset.__init__(self, root, classes, transform)

        classes_train = range(0, 23)
        classes_test = range(0, 23)

        '''
        if classes.start in classes_train:
            if classes.stop - 1 in classes_train:
                train = True

        if classes.start in classes_test:
            if classes.stop - 1 in classes_test:
                train = False
        '''
        if  dl_type == 'eval':
            train = False
        else:
            train = True

        with open(
            os.path.join(
            root,
            'HKVASIR_{}.txt'.format('train' if train else 'test')
            )
        ) as f:

            f.readline()
            index = 0
            nb_images = 0

            for (image_id, class_id, path) in map(str.split, f):
                nb_images += 1
                if int(class_id) - 1 in classes:
                    self.im_paths.append(os.path.join(root, path))
                    self.ys.append(int(class_id) - 1)
                    self.I += [index]
                    index += 1

            if train:
                print("!"*10)
                assert nb_images == type(self).nb_train_all
            else:
                assert nb_images == type(self).nb_test_all
