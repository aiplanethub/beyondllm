from enterprise_rag.fit_data import fit

data = fit("/home/adithya/Downloads/AdithyaHegde_Resume.pdf",chunk_size=100,chunk_overlap=10)

print(len(data),data)