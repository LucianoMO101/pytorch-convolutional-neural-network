# 🧠 Convolutional Neural Network (CNN) - Handgeschreven Cijfers (MNIST)

![MNIST Digits Example](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

Een **Convolutional Neural Network (CNN)** gebouwd met **PyTorch** in **Google Colab** om handgeschreven cijfers (0–9) te herkennen uit de **MNIST dataset**.  
Dit project is bedoeld als leerervaring om te begrijpen hoe CNN’s werken en hoe ze worden getraind om afbeeldingen te classificeren.

---

## 📘 Inhoudsopgave

1. [Wat is een CNN?](#-wat-is-een-convolutional-neural-network-cnn)
2. [Projectoverzicht](#-projectoverzicht)
3. [Modelarchitectuur](#-modelarchitectuur)
4. [Dataset](#-dataset)
5. [Benodigde installaties](#-benodigde-installaties)
6. [Uitvoeren in Google Colab](#-uitvoeren-in-google-colab)
7. [Resultaten](#-resultaten)
8. [Visuele Hulpmiddelen](#-visuele-hulpmiddelen)
9. [Toekomstige Verbeteringen](#-toekomstige-verbeteringen)
10. [Licentie](#-licentie)

---

## H1. 🧩 Wat is een Convolutional Neural Network (CNN)?

Een **Convolutional Neural Network (CNN)** is een type **neurale netwerkarchitectuur** dat speciaal ontworpen is om te werken met **beelddata**.  
In tegenstelling tot klassieke neurale netwerken, die elk pixel afzonderlijk behandelen, **scant** een CNN de afbeelding met **kleine filters (convoluties)** die patronen leren herkennen zoals randen, vormen en texturen.

Het netwerk leert stapsgewijs:
1. **Eenvoudige patronen** zoals lijnen en hoeken.  
2. **Complexere structuren** zoals delen van cijfers of letters.  
3. Uiteindelijk: **volledige objecten of cijfers**.

👉 Hierdoor presteren CNN’s uitzonderlijk goed bij taken zoals:
- Beeldherkenning  
- Gezichtsdetectie  
- Zelfrijdende auto’s  
- Medische beeldanalyse  

---

## H2. 💻 Projectoverzicht

In dit project trainen we een **CNN** om de **MNIST dataset** te classificeren — een collectie van 70.000 afbeeldingen van handgeschreven cijfers (0–9).  
Elke afbeelding is **28x28 pixels** en zwart-wit.

We gebruiken:
- **PyTorch** voor het bouwen en trainen van het model.  
- **Google Colab** als ontwikkelomgeving (met gratis GPU-ondersteuning).  
- **Matplotlib** om de resultaten te visualiseren.  

---

## H3. 🧱 Modelarchitectuur

Hieronder zie je de opbouw van het gebruikte model:

```text
Input (1x28x28) ─► Conv2d(1,6,3) ─► ReLU ─► MaxPool(2x2)
                ─► Conv2d(6,16,3) ─► ReLU ─► MaxPool(2x2)
                ─► Flatten (16*5*5)
                ─► Linear (400 → 120) ─► ReLU
                ─► Linear (120 → 84) ─► ReLU
                ─► Linear (84 → 10) ─► LogSoftmax
```
✨ Dit model bevat twee convolutionele lagen en drie volledig verbonden (dense) lagen.
Elke laag leert steeds complexere kenmerken van het beeld.

---

## H4. 🗂 Dataset
We gebruiken de **MNIST dataset**, beschikbaar in torchvision.datasets.MNIST.
Deze bevat:
- **60.000** trainingsafbeeldingen
- **10.000** testafbeeldingen
- Labels: cijfers **0–9**
De dataset wordt automatisch gedownload met:
```python
from torchvision import datasets, transforms
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='/cnn_data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='/cnn_data', train=False, download=True, transform=transform)
```
---
## H5. ⚙️ Benodigde installaties
Als je dit project lokaal wil draaien, zorg dat je de volgende pakketten hebt geïnstalleerd:
```bash
pip install torch torchvision matplotlib numpy pandas scikit-learn
```
Of gebruik **Google Colab**, waar alles standaard aanwezig is.

---
## H6. ▶️ Uitvoeren in Google Colab
1. Open een nieuwe Colab-notebook.
2. Upload de Python-code (cnn_mnist.py of jouw notebookversie).
3. Zorg dat je runtime op GPU staat (Runtime → Change runtime type → GPU).
4. Run alle cellen.

Tijdens training zie je de verliezen (loss) en nauwkeurigheid (accuracy) per epoch.
Na afloop worden de resultaten gevisualiseerd.

---

## H7. 📊 Resultaten
### ✅ Modelprestatie
Gemiddelde nauwkeurigheid na 5 epochs:
```
~98% op de testset 🎯
```
### 📈 Trainingsgrafieken
**Verlies per epoch**
<img width="572" height="457" alt="image" src="https://github.com/user-attachments/assets/a9c3863f-e186-42a4-91bc-248281a5e4d2" />

**Nauwkeurigheid per epoch**
<img width="567" height="462" alt="image" src="https://github.com/user-attachments/assets/27c952a2-e0ce-4966-92c7-98619999b8c6" />

