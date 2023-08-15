import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup


def AE_scores(model, imgs, chunk_size = 10000):
    
    n_objs = imgs.shape[0]
    n_chunks = int(np.ceil(n_objs / chunk_size))
    results = np.array([])
    print("%i chunks" % n_chunks)

    for i in range(n_chunks):
        print(i)
        imgs_chunk = np.expand_dims(imgs[chunk_size*i:(i+1)*chunk_size], axis = -1)
        preds = model.predict(imgs_chunk, batch_size = 512)

        scores = np.mean(np.square(imgs_chunk - preds), axis = (1,2)).reshape(-1)
        results = np.append(results, scores)


    return results


parser = input_options()
options = parser.parse_args()

f = h5py.File(options.fin, "r+")
labeler = tf.keras.models.load_model(options.labeler_name)
j1_images = f['j1_images'][:]
j1_scores = AE_scores(labeler, j1_images)
f.create_dataset('j1_AE_scores', data = j1_scores, compression = 'gzip')
del j1_images
j2_images = f['j2_images'][:]
j2_scores = AE_scores(labeler, j2_images)
f.create_dataset('j2_AE_scores', data = j2_scores, compression = 'gzip')
del j2_images
f.close()


