# -*- coding:utf-8 -*-
import tensorflow as tf

def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    defaults = [[0], [0.], [0.], [0.], [0.], ['']]
    Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species = tf.decode_csv(value, defaults)

    #因为使用的是鸢尾花数据集，这里需要对y值做转换
    preprocess_op = tf.case({
        tf.equal(Species, tf.constant('Iris-setosa')): lambda: tf.constant(0),
        tf.equal(Species, tf.constant('Iris-versicolor')): lambda: tf.constant(1),
        tf.equal(Species, tf.constant('Iris-virginica')): lambda: tf.constant(2),
    }, lambda: tf.constant(-1), exclusive=True)

    return tf.stack([SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]), preprocess_op

def create_pipeline(filename, batch_size, num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    example, label = read_data(file_queue)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    return example_batch, label_batch

x_train_batch, y_train_batch = create_pipeline('Iris-train.csv', 50, num_epochs=1000)
x_test, y_test = create_pipeline('Iris-test.csv', 60)

init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()  # local variables like epoch_num, batch_size
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(local_init_op)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Retrieve a single instance:
    try:
        #while not coord.should_stop():
        while True:
            example, label = sess.run([x_train_batch, y_train_batch])
            print (example)
            print (label)
    except tf.errors.OutOfRangeError:
        print ('Done reading')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
