# from    omniglot import Omniglot
# import  torchvision.transforms as transforms
# from    PIL import Image
import  os.path
import  numpy as np


class FUSENShot:

    def __init__(self, root, batchsz, n_way, k_shot, k_query, imgsz):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        self.resize = imgsz
        # x[0]: features, x[1]: labels
        self.x = [np.load(os.path.join(root, 'features_list.npy'), allow_pickle=True),
                  np.load(os.path.join(root, 'df_kinect_list.npy'), allow_pickle=True)]
        
        print('load from FUSE npy.')

        # [1623, 20, 84, 84, 1]
        # 1623 classes, written by 20 different users, 84*84 size, grey channel
        # TODO: can not shuffle here, we must keep training and test set distinct!
        
        # [39, each movement]
        # each_movement = [frames, 5, 14, 14] number of frames, channels, size, size 
        # User1-3: 1-10; User4 - 1-8, 10
        train_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28]
        test_index = [38]
#        all_index = list(np.linspace(0,38,39).astype('int'))
#        train_index = list(np.random.choice(all_index, 29, False))
#        test_index = list(set(all_index) - set(train_index))
        
        self.x_train, self.x_test = [self.x[0][train_index], self.x[1][train_index]], [self.x[0][test_index], self.x[1][test_index]]

        # self.normalization()

        self.batchsz = batchsz
        self.n_cls = len(self.x[0])  # 39
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        assert (k_shot + k_query) <= 400

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        # print("DB: train", self.x_train.shape, "test", self.x_test.shape)

        self.datasets_cache = {"train": self.load_data_cache_train(self.datasets["train"]),  # current epoch data cached
                               "test": self.load_data_cache_test(self.datasets["test"])}


    def load_data_cache_train(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, frames, 5, 14, 14]
        :return: A list with [support_set_x, support_set_y, query_x, query_y] ready to be fed to our networks
        """
        #  take 5 way 200 shot as example: 5 * 200
        setsz = self.k_shot * self.n_way # 200 * 5
        querysz = self.k_query * self.n_way # 200 * 5
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(1):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set
            # 10000 batchsz

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                # for each batch, we sample different selected_class
                # randomly chose n_way classes from all classes (data_pack.shape[0])
                
                selected_cls = np.random.choice(len(data_pack[0]), self.n_way, False)
                #print("training selected_cls is : ", selected_cls)
                # iterate the chosen classes
                for j, cur_class in enumerate(selected_cls):
                    # randomly chose (k_shot + k_query) images from x frames
                    selected_img = np.random.choice(data_pack[0][cur_class].shape[0], self.k_shot + self.k_query, False)

                    # meta-training and meta-test
                    # support set [0:k_shot], query set [k_shot:], if k_shot = 5, 0,1,2,3,4 support, 5-19 query
                    x_spt.append(data_pack[0][cur_class][selected_img[:self.k_shot]].transpose(0,3,1,2))
                    x_qry.append(data_pack[0][cur_class][selected_img[self.k_shot:]].transpose(0,3,1,2))
                    # y_spt be like [[0,0,0,0,0,...] [1,1,1,1,1,...] ...], total length = n_way, each small list will be k_shot length
                    # y_spt be like [[0,0,0,0,0,...] [1,1,1,1,1,...] ...], total length = n_way, each small list will be k_query length
                    y_spt.append(data_pack[1][cur_class][selected_img[:self.k_shot]])
                    y_qry.append(data_pack[1][cur_class][selected_img[self.k_shot:]])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 5, self.resize, self.resize)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot, 57)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 5, self.resize, self.resize)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query, 57)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                # b = batch_size
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)


            # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, 5, self.resize, self.resize)
            y_spts = np.array(y_spts).astype(np.float32).reshape(self.batchsz, setsz, 57)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, 5, self.resize, self.resize)
            y_qrys = np.array(y_qrys).astype(np.float32).reshape(self.batchsz, querysz, 57)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache
    
    def load_data_cache_test(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, frames, 5, 14, 14]
        :return: A list with [support_set_x, support_set_y, query_x, query_y] ready to be fed to our networks
        """
        #  take 5 way 200 shot as example: 5 * 200
        test_way = 1
        setsz = self.k_shot * test_way # 200 * 5
        querysz = self.k_query * test_way # 200 * 5
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set
            # 10000 batchsz

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                # for each batch, we sample different selected_class
                # randomly chose n_way classes from all classes (data_pack.shape[0])
                #print("len(data_pack[0]) is : ", len(data_pack[0]))
                selected_cls = np.random.choice(len(data_pack[0]), test_way, False)
                # iterate the chosen classes
                for j, cur_class in enumerate(selected_cls):
                    # randomly chose (k_shot + k_query) images from x frames
                    selected_img = np.random.choice(data_pack[0][cur_class].shape[0], self.k_shot + self.k_query, False)

                    # meta-training and meta-test
                    # support set [0:k_shot], query set [k_shot:], if k_shot = 5, 0,1,2,3,4 support, 5-19 query
                    x_spt.append(data_pack[0][cur_class][selected_img[:self.k_shot]].transpose(0,3,1,2))
                    x_qry.append(data_pack[0][cur_class][selected_img[self.k_shot:]].transpose(0,3,1,2))
                    # y_spt be like [[0,0,0,0,0,...] [1,1,1,1,1,...] ...], total length = n_way, each small list will be k_shot length
                    # y_spt be like [[0,0,0,0,0,...] [1,1,1,1,1,...] ...], total length = n_way, each small list will be k_query length
                    y_spt.append(data_pack[1][cur_class][selected_img[:self.k_shot]])
                    y_qry.append(data_pack[1][cur_class][selected_img[self.k_shot:]])

                # shuffle inside a batch
                perm = np.random.permutation(test_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(test_way * self.k_shot, 5, self.resize, self.resize)[perm]
                y_spt = np.array(y_spt).reshape(test_way * self.k_shot, 57)[perm]
                perm = np.random.permutation(test_way * self.k_query)
                x_qry = np.array(x_qry).reshape(test_way * self.k_query, 5, self.resize, self.resize)[perm]
                y_qry = np.array(y_qry).reshape(test_way * self.k_query, 57)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                # b = batch_size
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)


            # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, 5, self.resize, self.resize)
            y_spts = np.array(y_spts).astype(np.int).reshape(self.batchsz, setsz, 57)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, 5, self.resize, self.resize)
            y_qrys = np.array(y_qrys).astype(np.int).reshape(self.batchsz, querysz, 57)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache
    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            if mode is 'train':
                self.datasets_cache[mode] = self.load_data_cache_train(self.datasets[mode])
            elif mode is 'test':
                self.datasets_cache[mode] = self.load_data_cache_test(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch





if __name__ == '__main__':

    import  time
    import  torch
    # import  visdom

    # plt.ion()
    # viz = visdom.Visdom(env='omniglot_view')

    db = FUSENShot('FUSE', batchsz=20, n_way=5, k_shot=200, k_query=200, imgsz=14)

    for i in range(1000):
        x_spt, y_spt, x_qry, y_qry = db.next('train')


        # [b, setsz, h, w, c] => [b, setsz, c, w, h] => [b, setsz, 3c, w, h]
        x_spt = torch.from_numpy(x_spt)
        x_qry = torch.from_numpy(x_qry)
        y_spt = torch.from_numpy(y_spt)
        y_qry = torch.from_numpy(y_qry)
        batchsz, setsz, c, h, w = x_spt.size()


        # viz.images(x_spt[0], nrow=5, win='x_spt', opts=dict(title='x_spt'))
        # viz.images(x_qry[0], nrow=15, win='x_qry', opts=dict(title='x_qry'))
        # viz.text(str(y_spt[0]), win='y_spt', opts=dict(title='y_spt'))
        # viz.text(str(y_qry[0]), win='y_qry', opts=dict(title='y_qry'))


        time.sleep(10)

# FUSE normal sample
# train_index = [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28]
# test_index = [38]


# def concatenate_features(data):
    
#     temp = np.zeros((1,14,14,5)) 
    
#     for item in data:
#         temp = np.concatenate((temp, item))
    
#     temp = temp[1:]
#     return temp

# def concatenate_labels(data):
    
#     temp = np.zeros((1,57)) 
    
#     for item in data:
#         temp = np.concatenate((temp, item))
    
#     temp = temp[1:]
#     return temp

# data = np.load(os.path.join('FUSE', 'features_list.npy'), allow_pickle=True)

# labels = np.load(os.path.join('FUSE', 'df_kinect_list.npy'), allow_pickle=True)

# train_data, test_data, train_labels, test_labels = data[train_index], data[test_index], labels[train_index], labels[test_index]

# train_data, test_data, train_labels, test_labels = concatenate_features(train_data), concatenate_features(test_data), concatenate_labels(train_labels), concatenate_labels(test_labels)

# np.save('FUSE/bigUC_data.npy', train_data)
# np.save('FUSE/bigUC_labels.npy', train_labels)
# np.save('FUSE/smallUC_data.npy', test_data)
# np.save('FUSE/smallUC_labels.npy', test_labels)


