import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist= input_data.read_data_sets('MNIST_data', one_hot=True)

# Arbitrarily chosen, can be any size
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# 10 classes, 0-9
n_classes = 10

# Batch of features to read
batch_size = 100

# x = data, y = label
# 28x28 flattened to an array = 784
#                          height  width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def neural_network_model(input_data):
    hidden_layer_1 = {
        # tf.Variable with random values and [784, 500] shape.
        'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
        # tf.Variable with random values and [500] shape.
        'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))
    }
    hidden_layer_2 = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))
    }
    hidden_layer_3 = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))
    }
    output_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))
    }
    
    # input_data * weights + biases
    
    l1 = tf.add(tf.matmul(input_data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    
    # Activation function (threshold)
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    
    # Adam Optimizer learning_rate = 0.001 by default
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    n_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                
            print(f'Epoch {epoch} completed out of {n_epochs} (loss: {epoch_loss})')
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


if __name__ == '__main__':
    train_neural_network(x)
