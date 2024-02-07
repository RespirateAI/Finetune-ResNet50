import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()
import LARSOptimizer


total_steps = 10
learning_rate = 2.0
momentum = 0.9
weight_decay = 1e-4
temperature = 1.
num_classes = 4

def add_kd_loss(student_logits, teacher_logits, temperature):
  """Compute distillation loss."""
  teacher_probs = tf.nn.softmax(teacher_logits / temperature)
  kd_loss = tf.reduce_mean(temperature**2 * tf.nn.softmax_cross_entropy_with_logits(
      teacher_probs, student_logits / temperature))
  return kd_loss

# Define a small student ConvNet
def build_student_model():
  student_model = tf.keras.models.Sequential()
  student_model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(224, 224, 3)))
  student_model.add(tf.keras.layers.BatchNormalization())
  student_model.add(tf.keras.layers.Activation('relu'))
  student_model.add(tf.keras.layers.MaxPooling2D((4, 4)))
  student_model.add(tf.keras.layers.Conv2D(128, (3, 3)))
  student_model.add(tf.keras.layers.BatchNormalization())
  student_model.add(tf.keras.layers.Activation('relu'))
  student_model.add(tf.keras.layers.MaxPooling2D((4, 4)))
  student_model.add(tf.keras.layers.Conv2D(256, (3, 3)))
  student_model.add(tf.keras.layers.BatchNormalization())
  student_model.add(tf.keras.layers.Activation('relu'))
  student_model.add(tf.keras.layers.GlobalAveragePooling2D())
  student_model.add(tf.keras.layers.Dense(512, activation='relu'))
  student_model.add(tf.keras.layers.Dense(1000))
  return student_model

class Model(tf.keras.Model):
  def __init__(self, path):
    super(Model, self).__init__()
    self.teacher_saved_model = tf.saved_model.load(path)
    self.student_model = build_student_model()
    self.dense_layer = tf.keras.layers.Dense(units=num_classes, name="head_supervised_new")
    learning_rate_schedule = tf.keras.experimental.CosineDecay(learning_rate,
                                                               total_steps)
    self.optimizer = LARSOptimizer(
      learning_rate_schedule,
      momentum=momentum,
      weight_decay=weight_decay,
      exclude_from_weight_decay=['batch_normalization', 'bias'])

  def call(self, x):
    with tf.GradientTape() as tape:
      outputs = self.teacher_saved_model(x[0], trainable=False)
      teacher_logits_t = outputs['logits_sup']
      student_logits_t = self.student_model(x[0])
      loss_t = add_kd_loss(student_logits_t, teacher_logits_t, temperature)

      trainable_weights = self.student_model.trainable_weights
      print('Variables to train:', [v.name for v in trainable_weights])
      grads = tape.gradient(loss_t, trainable_weights)
      self.optimizer.apply_gradients(zip(grads, trainable_weights))
    return loss_t, x[0], teacher_logits_t, student_logits_t   