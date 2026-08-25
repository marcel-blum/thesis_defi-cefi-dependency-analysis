import re
import matplotlib.pyplot as plt

tex_path = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/8_deep_learning_model/Tables_Figures/tft_importance_combined.tex"
output_path = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/8_deep_learning_model/Tables_Figures/tft_encoder_decoder_hist.png"

with open(tex_path, "r") as f:
    content = f.read()


# Each numeric cell is either a plain decimal (e.g. "0.1234") or the
# rounding-threshold marker "$<$0.0001" emitted by fmt4() for near-zero
# importances; both forms must be matched so masked rows aren't silently
# dropped from the histogram.
NUM = r"(?:\$<\$0\.0001|[\d.]+)"
row_pattern = re.compile(
    rf"^\s*[\w\\_]+\s*&\s*({NUM})\s*&\s*({NUM})\s*&\s*{NUM}\s*\\\\",
    re.MULTILINE
)

def _parse_cell(s):
    return 0.0 if s.startswith(r"$<$") else float(s)

encoder, decoder = [], []
for enc_val, dec_val in row_pattern.findall(content):
    encoder.append(_parse_cell(enc_val))
    decoder.append(_parse_cell(dec_val))

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
bins = [i * 0.02 for i in range(26)]

axes[0].hist(encoder, bins=bins, color="black", alpha=0.75)
axes[0].set_title("Encoder", fontstyle='italic', fontsize=12)
axes[0].set_xlabel("Importance Weight", fontsize=11)
axes[0].set_ylabel("Frequency", fontsize=11)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

axes[1].hist(decoder, bins=bins, color="gray", alpha=0.75)
axes[1].set_title("decoder", fontstyle='italic', fontsize=12)
axes[1].set_xlabel("Importance Weight", fontsize=11)
axes[1].set_ylabel("Frequency", fontsize=11)
axes[1].tick_params(axis='y', labelleft=True)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()