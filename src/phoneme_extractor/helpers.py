import matplotlib.pyplot as plt
import torch


def plot_mel_phonemes(mel, compressed_frames, save_path=None):
    """
    Plot mel spectrogram with phoneme IDs overlaid directly on the spectrogram

    Args:
        mel: Mel spectrogram tensor [frames, mel_bins]
        compressed_frames: List of [phoneme_id, count] pairs representing phoneme alignment per frame
        save_path: Path to save the plot
    """
    assert mel.dim() == 2, f"Expected 2D mel tensor, got {mel.dim()}D"
    phn_frame_ids = [phoneme_id for phoneme_id, _ in compressed_frames]
    phn_frame_counts = [count for _, count in compressed_frames]

    # Create single plot - make it twice as wide
    fig, ax = plt.subplots(1, 1, figsize=(30, 8))

    # Convert mel to numpy for plotting
    mel_np = mel.cpu().numpy() if isinstance(mel, torch.Tensor) else mel

    # Add statistics to title instead of overlaying on spectrum
    unique_phonemes = len(set(phn_frame_ids))
    stats_text = f"Frames: {sum(phn_frame_counts)} | Unique Phonemes: {unique_phonemes} | Mel Bins: {mel.shape[1]}"
    title_text = f"Mel Spectrogram with Phoneme Alignment\n{stats_text}"

    # Plot mel spectrogram
    im = ax.imshow(
        mel_np.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_ylabel("Mel Bins")
    ax.set_xlabel("Frame Index")
    ax.set_title(title_text)
    plt.colorbar(im, ax=ax, label="Magnitude")

    # Overlay phoneme information
    frame_pos = 0
    for phn_id, count in zip(phn_frame_ids, phn_frame_counts):
        # Draw vertical boundary lines (except for first segment)
        if frame_pos > 0:
            ax.axvline(
                x=frame_pos - 0.5,
                color="red",
                linestyle="-",
                alpha=0.8,
                linewidth=2,
            )

        # Add phoneme ID text at the top of the spectrogram
        if count > 1:  # Only add text if segment is wide enough
            text_x = frame_pos + count / 2
            text_y = mel.shape[1] - 2  # Near the top of the mel bins

            # Add text with background for visibility
            ax.text(
                text_x,
                text_y,
                str(phn_id),
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8),
            )

        frame_pos += count

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.tight_layout()
    plt.show()
    return
