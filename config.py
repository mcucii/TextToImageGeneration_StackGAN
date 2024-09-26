DATASET_NAME = 'birds'
EMBEDDING_TYPE = 'cnn-rnn'

DEVICE = 'cpu'

NET_G = ''  # putanja do fajla sa težinama modela
NET_D = ''
STAGE1_G = ''
STAGE1_D = ''
#DATA_DIR =  '/content/drive/MyDrive/RIProject/data_reduced20/birds'
DATA_DIR =  '/home/masa/Desktop/MindArtify/data_reduced1/birds'

IMG_SIZE = 64
STAGE = 1

TRAIN = True
TRAIN_BATCH_SIZE = 32
TRAIN_MAX_EPOCH = 100

TRAIN_PRETRAINED_MODEL = ''
TRAIN_PRETRAINED_EPOCH = 600

TRAIN_LR_DECAY_EPOCH = 35    # 35 - posle koliko epoha se koeficijent ucenja smanjuje

TRAIN_DISCRIMINATOR_LR = 0.0002
TRAIN_GENERATOR_LR = 0.0002


# Modal options

GAN_CONDITION_DIM = 128
GAN_DF_DIM = 96 #96
GAN_GF_DIM = 192 #192
Z_DIM = 100 # 100


TEXT_DIMENSION = 1024

EMBEDDING_FILENAME = "char-CNN-RNN-embeddings.pickle"
