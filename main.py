from dataset import create_dataset
from preprocess import preprocess_image
from model import Model
import numpy as np

total_steps = 10

def _preprocess(x):
  print("WWW",x)
  x[0] = preprocess_image(
      x[0], 224, 224, is_training=True, color_distort=False)
  return x

ds = create_dataset().prefetch(100).batch(16)
model = Model("gs://simclr-checkpoints-tf2/simclrv2/finetuned_100pct/r50_1x_sk0/saved_model/")

iterator = iter(ds)
for it in range(total_steps):
  x = next(iterator)
  loss, image, teacher_logits, student_logits = model(x)
  teacher_logits = teacher_logits.numpy()
  student_logits = student_logits.numpy()
  labels = teacher_logits.argmax(-1)
  pred = student_logits.argmax(-1)
  correct = np.sum(pred == labels)
  total = labels.size
  print("[Iter {}] Loss: {} Top 1: {}".format(it+1, loss, correct/float(total)))