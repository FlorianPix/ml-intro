import tensorflow as tf
import models
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

training_losses = []
training_accuracies = []
test_losses = []
test_accuracies = []
layers = range(1, 4)
epochs = range(2, 15)
xs = []
ys = []

for number_of_epochs in epochs:
    for number_of_layers in layers:
        model = models.multiple_layer_model(number_of_layers)

        model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=number_of_epochs)

        '''
        fig, ax = plt.subplots()
        ax.plot(history.epoch, history.history["loss"], linewidth=2.0)
        ax.plot(history.epoch, history.history["accuracy"], linewidth=2.0)
        plt.show()
        '''

        training_loss = history.history["loss"][-1]
        training_accuracy = history.history["accuracy"][-1]
        test_loss, test_accuracy = model.evaluate(x_test,  y_test, verbose=2)

        training_losses.append(training_loss)
        training_accuracies.append(training_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        xs.append(number_of_layers)
        ys.append(number_of_epochs)

'''
fig, ax = plt.subplots()
xs = [*layers]
ax.plot(xs, training_losses, linewidth=2.0)
ax.plot(xs, test_losses, linewidth=2.0)
plt.show()
'''

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
tra = ax.scatter(xs, ys, training_accuracies)
tea = ax.scatter(xs, ys, test_accuracies)
tra.set_label('training accuracy')
tea.set_label('test accuracy')
ax.set_xlabel('number of layers')
ax.set_ylabel('epochs')
ax.set_zlabel('accuracy')
ax.legend()
plt.show()
