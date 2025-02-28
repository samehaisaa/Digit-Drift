import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_latent_space(encoder, x_test, y_test, decoder):
    latent_test = encoder.predict(x_test)
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for digit in range(10):
        mask = y_test == digit
        ax.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                   color=colors[digit], alpha=0.5, label=str(digit))
    
    ax.set_title("2D PCA of Latent Representations")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Digit", loc="upper right")
    
    def onclick(event):
        if event.inaxes == ax:
            click_x, click_y = event.xdata, event.ydata
            distances = np.linalg.norm(latent_2d - np.array([click_x, click_y]), axis=1)
            closest_idx = np.argmin(distances)
            closest_latent = latent_test[closest_idx:closest_idx+1]
            reconstructed_img = decoder.predict(closest_latent)
            fig_rec, ax_rec = plt.subplots()
            ax_rec.imshow(reconstructed_img.squeeze(), cmap='gray')
            ax_rec.set_title(f"Reconstructed Digit: {y_test[closest_idx]}")
            ax_rec.axis("off")
            plt.show()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
