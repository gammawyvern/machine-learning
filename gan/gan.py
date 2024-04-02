from matplotlib import pyplot;

from numpy.random import rand, randn;
from numpy import hstack, zeros, ones;

from keras.models import Sequential;
from keras.layers import Dense;

########################################
# Generation Code
########################################

def generate_real_samples(n):
    x1 = rand(n) - 0.5;
    x2 = 2**x;
    x1 = x1.reshape(n, 1);
    x2 = x2.reshape(n, 1);
    x = hstack((x1, x2));
    y = ones((n, 1));
    return x, y;

########################################
# Discriminator Code
########################################

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

########################################
# Generator Code
########################################

def define_generator(latent_dim, n_outputs=2):
    model = Sequential();
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim));
    model.add(Dense(n_outputs, activation='linear'));
    return model;

def generate_latent_points(latent_dim, n):
    x_input = randn(latent_dim * n);
    x_input = x_input.reshape(n, latent_dim);
    return x_input;

def generate_fake_samples(generator, latent_dim, n):
    x_input = generate_latent_points(latent_dim, n);
    x = generator.predict(x_input);
    y = zeros((n, 1));
    return x, y

########################################
# GAN Code
########################################

def define_gan(generator, descriminator):
    descriminator.trainable = False;
    model = Sequential();
    model.add(generator);
    model.add(descriminator);
    model.compile(loss='binary_crossentropy', optimizer='adam');
    return model;

def train_gan(gan_model, latent_dim, n_epochs=10000, n_batch=128):
    for n in range(n_epochs):
        x_gan = generate_latent_points(latent_dim, n_batch);
        y_gan = ones((n_batch, 1));
        gan_model.train_on_batch(x_gan, y_gan);

def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=128, n_eval=2000):
    half_batch = int(n_batch / 2);

    for i in range(n_epochs):
        x_real, y_real = generate_real_samples(half_batch);
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch);
        d_model.train_on_batch(x_real, y_real);
        d_model.train_on_batch(x_fake, y_fake);
        x_gan = generate_latent_points(latent_dim, n_batch);
        y_gan = ones((n_batch, 1));
        gan_model.train_on_batch(x_gan, y_gan);

        if (i+1) % n_eval == 0:
            summarize_performance(i, g_model, d_model, latent_dim);

########################################
# Performance Checking Code
########################################

def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
    x_real, y_real = generate_real_samples(n);
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0);
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n);
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0);
    print(epoch, acc_real, acc_fake);
    pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
    pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
    pyplot.show()

########################################
# Testing/Running Code
########################################

latent_dim = 5;
discriminator = define_discriminator();
generator = define_generator(latent_dim);
gan_model = define_gan(generator, discriminator);
train(generator, discriminator, gan_model, latent_dim)

