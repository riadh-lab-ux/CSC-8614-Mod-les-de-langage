from transformers import GPT2Model

model = GPT2Model.from_pretrained("gpt2")

# récupérer la matrice des embeddings positionnels
position_embeddings = model.wpe.weight

print("Shape position embeddings:", position_embeddings.size())

# infos de config utiles
print("n_embd:", model.config.n_embd)
print("n_positions:", model.config.n_positions)

import plotly.express as px
from sklearn.decomposition import PCA

# extraire les 50 premières positions (tensor -> numpy)
positions = position_embeddings[:200].detach().cpu().numpy()

pca = PCA(n_components=2)
reduced = pca.fit_transform(positions)

fig = px.scatter(
    x=reduced[:, 0],
    y=reduced[:, 1],
    text=[str(i) for i in range(len(reduced))],
    color=list(range(len(reduced))),
    title="Encodages positionnels GPT-2 (PCA, positions 0-200)",
    labels={"x": "PCA 1", "y": "PCA 2"}
)

# sauver dans TP1/positions_50.html
fig.write_html("../positions_200.html")
print("Saved: TP1/positions_200.html")
