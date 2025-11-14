//the literal file was >25Mb to push into github,so i transfferred code here,sorry for the inconvenience

import os
import random
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from IPython.display import Audio,display
from tqdm import tqdm

class TrainAudioSpectrogramDataset(Dataset):
    """
    This dataset class is reused directly. It loads audio from subfolders,
    creates a log-mel-spectrogram, and provides a one-hot encoded label.
    This is exactly what our Conditional GAN needs.
    """
    def __init__(self, root_dir, categories, max_frames=512, fraction=1.0):
        self.root_dir = root_dir
        self.categories = categories
        self.max_frames = max_frames
        self.file_list = []
        self.class_to_idx = {cat: i for i, cat in enumerate(categories)}

        for cat_name in self.categories:
            cat_dir = os.path.join(root_dir, cat_name)
            files_in_cat = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.endswith(".wav")]
            num_to_sample = int(len(files_in_cat) * fraction)
            sampled_files = random.sample(files_in_cat, num_to_sample)
            label_idx = self.class_to_idx[cat_name]
            self.file_list.extend([(file_path, label_idx) for file_path in sampled_files])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=256, n_mels=128
        )(wav)
        log_spec = torch.log1p(mel_spec)
        log_spec = (log_spec - log_spec.mean()) / (log_spec.std() + 1e-6)


        _, _, n_frames = log_spec.shape
        if n_frames < self.max_frames:
            pad = self.max_frames - n_frames
            log_spec = F.pad(log_spec, (0, pad))
        else:
            log_spec = log_spec[:, :, :self.max_frames]

        label_vec = F.one_hot(torch.tensor(label), num_classes=len(self.categories)).float()
        return log_spec, label_vec

# ==============================================================================
# 2. GAN MODEL DEFINITIONS (GENERATOR & DISCRIMINATOR)
# ==============================================================================

class CGAN_Generator(nn.Module):
    """ The Forger/Artist """
    def __init__(self, latent_dim, num_categories, spec_shape=(128, 512)):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_categories = num_categories
        self.spec_shape = spec_shape

        # Upsampling architecture
        self.fc = nn.Linear(latent_dim + num_categories, 256 * 8 * 32)
        self.unflatten_shape = (256, 8, 32) # (channels, H, W)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> 16x64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> 32x128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> 64x256
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1), # -> 128x512
            nn.Tanh() # Use ReLU to match the log1p output range [0, inf)
        )

    def forward(self, z, y):
        # Concatenate noise vector (z) and label (y)
        h = torch.cat([z, y], dim=1)
        h = self.fc(h)
        h = h.view(-1, *self.unflatten_shape)
        fake_spec = self.net(h)
        return fake_spec

class CGAN_Discriminator(nn.Module):
    """ The Detective/Critic """
    def __init__(self, num_categories, spec_shape=(128, 512)):
        super().__init__()
        self.num_categories = num_categories
        self.spec_shape = spec_shape
        H, W = spec_shape

        # Embedding for the label to match the image dimensions
        self.label_embedding = nn.Linear(num_categories, H * W)

        # Downsampling architecture
        self.net = nn.Sequential(
            # Input channel is 2: 1 for spectrogram, 1 for label map
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1), # -> 64x256
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> 32x128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> 16x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> 8x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Final output layer to produce a single logit
            nn.Conv2d(256, 1, kernel_size=(8, 32), stride=1, padding=0) # -> 1x1
        )

    def forward(self, spec, y):
        # Create a channel for the label and concatenate it with the spectrogram
        label_map = self.label_embedding(y).view(-1, 1, *self.spec_shape)
        h = torch.cat([spec, label_map], dim=1)

        # Pass through the network
        logit = self.net(h)
        return logit.view(-1, 1) # Flatten to a single value per item in batch

# ==============================================================================
# 3. UTILITY FUNCTIONS (GENERATION, SAVING)
# ==============================================================================

def generate_audio_gan(generator, category_idx, num_samples, device, sample_rate=22050):
    generator.eval()
    num_categories = generator.num_categories
    latent_dim = generator.latent_dim

    # Prepare label and noise
    y = F.one_hot(torch.tensor([category_idx]), num_classes=num_categories).float().to(device)
    if num_samples > 1:
        y = y.repeat(num_samples, 1)   # expand label to match z batch
    z = torch.randn(num_samples, latent_dim, device=device)

    with torch.no_grad():
        log_spec_gen = generator(z, y)   # shape (B,1,n_mels,frames)

    # generator uses Tanh -> outputs in [-1,1]
    # Convert from [-1,1] -> [0,1] (undo Tanh range)
    log_spec_gen = (log_spec_gen + 1.0) / 2.0   # now in [0,1]

    # Now convert back to mel magnitudes (undo the log1p and min-max)
    # We used per-sample min-max during dataset preprocessing, so we
    # must map [0,1] -> original log-mel scale before expm1.
    # Without storing per-sample min/max, we can assume the original log-mel
    # values were >= 0 and use a simple scaling factor (preferable: store mins/max per dataset)
    # Here we'll map [0,1] -> an empirical range, e.g., [0, max_logmel]
    # A conservative approach: scale to [0, 10]. Adjust if your dataset uses a different range.
    max_logmel = 10.0
    log_spec_gen = log_spec_gen * max_logmel  # now approx in [0, max_logmel]

    # Clamp for safety
    log_spec_gen = torch.clamp(log_spec_gen, min=0.0, max=max_logmel)

    # Convert spectrogram back to audio
    spec_gen = torch.expm1(log_spec_gen)   # invert log1p -> mel magnitudes
    spec_gen = spec_gen.squeeze(1)         # (B, n_mels, frames)




    import librosa

# Process batch-wise
    waveforms = []
    for i in range(spec_gen.size(0)):
        mel_spec_np = spec_gen[i].cpu().numpy()  # shape (n_mels, frames)
    # Convert mel spectrogram to linear spectrogram
        linear_spec_np = librosa.feature.inverse.mel_to_stft(mel_spec_np, sr=sample_rate, n_fft=1024)
    # Convert to torch tensor
        linear_spec = torch.from_numpy(linear_spec_np).unsqueeze(0).to(device)  # shape (1, n_fft//2+1, frames)
    # Griffin-Lim to get waveform
        waveform = torchaudio.transforms.GriffinLim(n_fft=1024, hop_length=256, win_length=1024, n_iter=32).to(device)(linear_spec)
    # Normalize waveform
        waveform = waveform / (waveform.abs().amax() + 1e-8)
        waveforms.append(waveform)

    waveforms = torch.stack(waveforms, dim=0)  # shape (B, time)
    return waveforms.cpu()


    griffin = torchaudio.transforms.GriffinLim(
        n_fft=1024, hop_length=256, win_length=1024, n_iter=32
    ).to(device)

    waveform = griffin(linear_spec)  # (B, time)
    # Normalize each waveform to -1..1
    waveform = waveform / (waveform.abs().amax(dim=1, keepdim=True) + 1e-8)
    return waveform.cpu()  # shape (B, time)

def save_and_play(wav, sample_rate, filename):
    if wav.dim() > 2: wav = wav.squeeze(0)
    torchaudio.save(filename, wav, sample_rate=sample_rate)
    print(f"Saved to {filename}")
    display(Audio(data=wav.numpy(), rate=sample_rate))

# ==============================================================================
# 4. GAN TRAINING FUNCTION
# ==============================================================================

def train_gan(generator, discriminator, dataloader, device, categories, epochs, lr, latent_dim):
    # Optimizers for each model
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Create directories for output
    os.makedirs("gan_generated_audio", exist_ok=True)
    os.makedirs("gan_spectrogram_plots", exist_ok=True)

    for epoch in range(1, epochs + 1):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for real_specs, labels in loop:
            real_specs = real_specs.to(device)
            labels = labels.to(device)
            batch_size = real_specs.size(0)

            # Labels for loss calculation
            real_labels_tensor = torch.ones(batch_size, 1, device=device)
            fake_labels_tensor = torch.zeros(batch_size, 1, device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            real_output = discriminator(real_specs, labels)
            loss_D_real = criterion(real_output, real_labels_tensor)

            z = torch.randn(batch_size, latent_dim, device=device)
            fake_specs = generator(z, labels)

            fake_output = discriminator(fake_specs.detach(), labels)
            loss_D_fake = criterion(fake_output, fake_labels_tensor)

            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            output = discriminator(fake_specs, labels)
            loss_G = criterion(output, real_labels_tensor)

            loss_G.backward()
            optimizer_G.step()

            loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

        # --- End of Epoch: Generate and save samples ---
        if epoch % 1 == 0:
            print(f"\n--- Generating Samples for Epoch {epoch} ---")
            generator.eval()

            # --- PLOTTING CODE THAT WAS MISSING ---
            fig, axes = plt.subplots(1, len(categories), figsize=(4 * len(categories), 4))
            if len(categories) == 1: axes = [axes] # Make it iterable

            for cat_idx, cat_name in enumerate(categories):
                y_cond = F.one_hot(torch.tensor([cat_idx]), num_classes=generator.num_categories).float().to(device)
                z_sample = torch.randn(1, generator.latent_dim).to(device)
                with torch.no_grad():
                    spec_gen_log = generator(z_sample, y_cond)

                spec_gen_log_np = spec_gen_log.squeeze().cpu().numpy()
                axes[cat_idx].imshow(spec_gen_log_np, aspect='auto', origin='lower', cmap='viridis')
                axes[cat_idx].set_title(f'{cat_name} (Epoch {epoch})')
                axes[cat_idx].axis('off')

            plt.tight_layout()
            plt.savefig(f'gan_spectrogram_plots/epoch_{epoch:03d}.png')
            plt.show()
            plt.close(fig) # Close the figure to free up memory
            # --- END OF PLOTTING CODE ---

            # --- Audio generation (was already here) ---
            for cat_idx, cat_name in enumerate(categories):
                wav = generate_audio_gan(generator, cat_idx, 1, device)
                fname = f"gan_generated_audio/{cat_name}_ep{epoch}.wav"
                save_and_play(wav, sample_rate=22050, filename=fname)

            generator.train() # Set back to training mode
            print("--- End of Sample Generation ---\n")

if __name__ == '__main__':
    # --- Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LATENT_DIM = 100 # Standard for GANs
    EPOCHS = 200 # GANs often require more epochs
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-4 # Common learning rate for GANs with Adam

    # --- Paths and Data Setup ---
    BASE_PATH = "/Users/ramupadhyay/Desktop/"
    TRAIN_PATH = os.path.join(BASE_PATH, 'train')
    train_categories = sorted([d for d in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, d))])
    NUM_CATEGORIES = len(train_categories)

    print(f"Using device: {DEVICE}")
    print(f"Found {NUM_CATEGORIES} categories: {train_categories}")

    train_dataset = TrainAudioSpectrogramDataset(
        root_dir=TRAIN_PATH, categories=train_categories
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # --- Initialize Models ---
    generator = CGAN_Generator(LATENT_DIM, NUM_CATEGORIES).to(DEVICE)
    discriminator = CGAN_Discriminator(NUM_CATEGORIES).to(DEVICE)

    # --- Start Training ---
    # Compute mean and std from a small subset of your training spectrograms
    spec_mean, spec_std = 0, 0
    count = 0

    for batch, _ in train_loader:
        spec_mean += batch.mean()
        spec_std += batch.std()
        count += 1
        if count > 10:  # don't loop through all batches if dataset is big
         break

    spec_mean /= count
    spec_std /= count

    print(f"Spectrogram mean: {spec_mean:.4f}, std: {spec_std:.4f}")

    train_gan(
        generator=generator,
        discriminator=discriminator,
        dataloader=train_loader,
        device=DEVICE,
        categories=train_categories,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        latent_dim=LATENT_DIM
    )
