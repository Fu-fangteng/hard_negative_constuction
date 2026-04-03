from angle_emb import AnglE
from angle_emb.utils import cosine_similarity

# Load model
angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

# Encode documents
doc_vecs = angle.encode([
    'The weather is great!',
    'The weather is very good!',
    'i am going to bed'
])

# Calculate pairwise similarity
for i, dv1 in enumerate(doc_vecs):
    for dv2 in doc_vecs[i+1:]:
        print(cosine_similarity(dv1, dv2))