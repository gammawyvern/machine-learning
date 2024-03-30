from matplotlib import pyplot;

from numpy.random import rand, randn;
from numpy import hstack, zeros, ones;

from keras.models import Sequential;
from keras.layers import Dense;

def func(x):
    return x*x;

def generate_real_samples(n):
    x1 = rand(n) - 0.5;
    x2 = x1 * x1;
    x1 = x1.reshape(n, 1);
    x2 = x2.reshape(n, 1);
    x = hstack((x1, x2));
    y = ones((n, 1));
    return x, y;

def generate_fake_samples(n):
    x1 = -1 + rand(n) * 2;
    x2 = -1 + rand(n) * 2;
    x1 = x1.reshape(n, 1);
    x2 = x2.reshape(n, 1);
    x = hstack((x1, x2));
    y = zeros((n, 1));
    return x, y;

def define_discriminator(n_inputs=2):
    model = Sequential();
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs));
    model.add(Dense(1, activation='sigmoid'));
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']);
    return model;

def train_discriminator(model, n_epochs=1000, n_batch=128):
    half_batch = int(n_batch / 2);
    for i in range(n_epochs):
        x_real, y_real = generate_real_samples(half_batch);
        model.train_on_batch(x_real, y_real);
        x_fake, y_fake = generate_fake_samples(half_batch);
        model.train_on_batch(x_fake, y_fake);
        _, acc_real = model.evaluate(x_real, y_real, verbose=0);
        _, acc_fake = model.evaluate(x_fake, y_fake, verbose=0);
        print(i, acc_real, acc_fake);

model = define_discriminator();
train_discriminator(model);

# Generate data code
# data = generate_samples();
# pyplot.scatter(data[:, 0], data[:, 1]);
# pyplot.show();


# Display data code
# inputs = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
# outputs = [func(x) for x in inputs];
#
# pyplot.plot(inputs, outputs);
# pyplot.show();

