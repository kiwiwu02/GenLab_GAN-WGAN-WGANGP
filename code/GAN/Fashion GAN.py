import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tqdm import tqdm
import time
import os


os.makedirs("outputs", exist_ok=True)


(train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 127.5 - 1.0
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(1000).batch(128).cache().prefetch(
    tf.data.AUTOTUNE)



def build_generator(latent_dim=100):
    model = tf.keras.Sequential([
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(latent_dim,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, 5, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(1, 5, strides=2, padding='same', use_bias=False, activation='tanh')
    ])
    return model


def build_discriminator(is_wgan=False):
    model = tf.keras.Sequential([
        layers.Conv2D(64, 5, strides=2, padding='same', input_shape=(28, 28, 1)),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, 5, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation=None if is_wgan else 'sigmoid')
    ])
    return model



class GANTrainer:
    def __init__(self, model_type="gan", latent_dim=100, lr=2e-4, clip_value=0.01, lambda_gp=10):
        self.model_type = model_type
        self.generator = build_generator(latent_dim)
        self.discriminator = build_discriminator(is_wgan=(model_type != "gan"))
        self.opt_gen = tf.keras.optimizers.Adam(lr, beta_1=0.5)
        self.opt_disc = tf.keras.optimizers.Adam(lr, beta_1=0.5)
        self.latent_dim = latent_dim
        self.clip_value = clip_value
        self.lambda_gp = lambda_gp
        self.history = {'g_loss': [], 'd_loss': [], 'diversity': []}
        self.model_name = model_type.upper()

    def gradient_penalty(self, real_images, fake_images):
        alpha = tf.random.uniform([len(real_images), 1, 1, 1], 0., 1.)
        interpolates = alpha * real_images + (1 - alpha) * fake_images
        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            pred = self.discriminator(interpolates)
        gradients = tape.gradient(pred, interpolates)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        return tf.reduce_mean(tf.square(slopes - 1.))

    def train_step(self, images):
        noise = tf.random.normal([images.shape[0], self.latent_dim])

        for _ in range(3 if self.model_type != "gan" else 1):
            with tf.GradientTape() as tape_d:
                fake_images = self.generator(noise, training=False)
                if self.model_type == "gan":
                    d_real = self.discriminator(images, training=True)
                    d_fake = self.discriminator(fake_images, training=True)
                    real_loss = tf.reduce_mean(
                        tf.keras.losses.binary_crossentropy(tf.ones_like(d_real), d_real))
                    fake_loss = tf.reduce_mean(
                        tf.keras.losses.binary_crossentropy(tf.zeros_like(d_fake), d_fake))
                    d_loss = (real_loss + fake_loss) / 2
                else:
                    d_loss = tf.reduce_mean(self.discriminator(fake_images)) - tf.reduce_mean(
                        self.discriminator(images))
                    if self.model_type == "wgan-gp":
                        d_loss += self.lambda_gp * self.gradient_penalty(images, fake_images)

            d_grads = tape_d.gradient(d_loss, self.discriminator.trainable_variables)
            self.opt_disc.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

            if self.model_type == "wgan":
                for var in self.discriminator.trainable_variables:
                    var.assign(tf.clip_by_value(var, -self.clip_value, self.clip_value))

        with tf.GradientTape() as tape_g:
            fake_images = self.generator(noise, training=True)
            if self.model_type == "gan":
                g_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(tf.ones_like(self.discriminator(fake_images)),
                                                        self.discriminator(fake_images)))
            else:
                g_loss = -tf.reduce_mean(self.discriminator(fake_images))

        g_grads = tape_g.gradient(g_loss, self.generator.trainable_variables)
        self.opt_gen.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return float(g_loss), float(d_loss)

    def calculate_diversity(self, num=100):
        samples = self.generator(tf.random.normal([num, self.latent_dim])).numpy()
        return float(np.mean(np.std(samples, axis=0)))

    def train(self, epochs=15):
        fixed_noise = tf.random.normal([25, self.latent_dim])
        start_time = time.time()

        for epoch in tqdm(range(epochs), desc=f"{self.model_name} Training"):
            g_losses, d_losses = [], []

            # 加一层 batch 进度条
            batch_bar = tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for batch in batch_bar:
                g_loss, d_loss = self.train_step(batch)
                g_losses.append(g_loss)
                d_losses.append(d_loss)
                batch_bar.set_postfix({"g_loss": f"{g_loss:.4f}", "d_loss": f"{d_loss:.4f}"})

            self.history['g_loss'].append(np.mean(g_losses))
            self.history['d_loss'].append(np.mean(d_losses))
            self.history['diversity'].append(self.calculate_diversity())

            if (epoch % 5 == 0) or (epoch == epochs - 1):
                self.save_generated_images(fixed_noise, epoch)

        self.save_final_report()

    def save_generated_images(self, noise, epoch):
        generated = self.generator(noise, training=False).numpy()
        fig, axes = plt.subplots(5, 5, figsize=(6, 6))
        for ax, img in zip(axes.flatten(), generated):
            ax.imshow(img.squeeze(), cmap='gray', vmin=-1, vmax=1)
            ax.axis('off')
        plt.suptitle(f"{self.model_name} | Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(f"outputs/{self.model_name}_epoch{epoch}.png", dpi=300)
        plt.close()

    def save_final_report(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['g_loss'], label='Generator Loss')
        plt.plot(self.history['d_loss'], label='Discriminator Loss')
        plt.title(f"{self.model_name} - Loss Curve")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['diversity'], 'r-o')
        plt.title(f"{self.model_name} - Diversity Curve")
        plt.xlabel('Epoch')
        plt.ylabel('Pixel STD')

        plt.tight_layout()
        plt.savefig(f"outputs/{self.model_name}_final_report.png", dpi=300)
        plt.close()



def train_gan():
    gan = GANTrainer(model_type="gan")
    print("\n=== Training GAN ===")
    gan.train(epochs=15)


def train_wgan():
    wgan = GANTrainer(model_type="wgan")
    print("\n=== Training WGAN ===")
    wgan.train(epochs=15)


def train_wgan_gp():
    wgan_gp = GANTrainer(model_type="wgan-gp", lambda_gp=10)
    print("\n=== Training WGAN-GP ===")
    wgan_gp.train(epochs=15)




train_gan()
# train_wgan()
#train_wgan_gp()
