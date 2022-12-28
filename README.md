# Painting-Retrieval
Romanian Painting retrieval on Pandora 18k and PandoraRom using MobileNet

See only: training.ipynb  + PaintingRetrieval.ipynb

## Definirea problemei

Problemele de *image retrieval* se referă la găsirea și returnarea unor imagini similare dintr-o baza de date de imagini. Tehnicile folosite în trecut includeau folosirea unor metadate cu descrierea imaginilior pentru a facilita cautarea, proces care poate dura foarte mult in cazurile in care este descrierea este facuta manual.
O data cu avansarea tehnologiei s-au căutat tehnici automate de *image retrieval*, una din ele fiind cea prezentată in acest raport, cea bazată pe rețele neuronale adânci. 

**Scopul final** este realizarea unui sistem de *image retrieval* pe o baza de date cu tablouri artistice. Mai exact, vom avea ca input un tablou care aparține unui curent artistic și vom căuta și returna tablouri ale artiștilor români care se aseamăna cât mai mult în stil și compozitie cu imaginea de la intrare.  

![[Rom_Impress.png]]


## Specificul bazei de date - Pandora 18K

Pandora 18K contine imagini cu diferite opere de arta impartite in 18 clase si per autor. Ea a fost creata pentru proiectul *Artistic Movement Recognition by Boosted Fusion of Color Structure and Topographic Description* [1] de catre Corneliu Florea și Cosmin Țoca din cadrul laboratorului de Image Processing al Universității Politehnica. Principala sursă a imaginilor a fost *Wikiart*, dar 25% au fost extrase din alte surse. De asemenea, s-a căutat balansarea numărului total de imagin din fiecare clasă. După colectarea imagilor si împărțirea in cele 18 stitluri de artă alese întreaga bază de date a fost analizată de un expert în arte, iar imaginile considerate non-artistice au fost înlăturate. În plus, etichetele multiple au fost înlăturate, doar cele dominante rămânând. 
O  diferență față de alte baza de date concentrate pe curente artisitice este ca aici un autor nu aparține exclusiv unui curent artistic. Un exemplu este Picasso care a pictat lucrări expresioniste, dar si surealiste sau care aparțin de cubism. [1]
![[ClaselePandora.png]]

Pentru ca in fiecare clasa se afla foldere separate pentru autori primul lucru pe care l-am facut a fost sa unesc toate imaginile astfel incat fiecare clasa sa aiba o grupare de poze. 
![[numberOfImagesPerClass.png]]

## Arhictectura - MobileNet
![[MobileNEt.png]]

MobileNet este o rețea covoluțională creată mai mult pentru aplicații mobile, sau aplicații în care puterea de procesare a mai limitată.

MobileNet folosește așa numita "*depthwise separable convolution*". Acestea reduc semnificativ numărul de parametrii comparativ cu o rețea cu același număr de layere dar care folosește convoluția normală, astfel rezultând o rețea cu mult mai rapidă.

*Depthwise separable convolution* a pornit de la ideea că adâncimea unui filtru și dimensiunea spațială pot fi separate. Astfel, depthwise separable convolution este formată din două părți:

1. **Depthwise convolution** = aplică un filtru diferit pe fiecare canal de la intrare
2. **Pointwise convolution** = o convoluție 1x1, de adâncime egala cu numărul de canale, efectuată pentru a modifica dimensiunea ieșirii

![[depthwise.png]]

Diferența majoră dintre MobileNet și o arhitectură CNN clasică este că la un CNN clasic, straturile de batch normalization si ReLu urmează imediat după un strat de convoluție. La MobileNet insă, layerul de convoluție fiind împărțit vom avea de doua ori mai multe straturi de BN si ReLu.

![[Pasted image 20221217161736.png]]

### MobileNetV2

MobileNetV2 aduce o îmbunătățire față de rețeaua inițială, și anume blocurile de "**inverted residual structure**."

![[Pasted image 20221217162122.png]]

## Libraria - Pytorch

PyTorch este o librărie open source pentru machine learning, destinată în special creârii și antrenării rețelelor neuronale. A fost creată de grupul de cercetători de la Facebook și poate fi folosită atât in Python cât și în C++.

PyTorch definește o clasă numită Tensor (`torch.Tensor`) pentru a stoca și opera vectori multidimensionali. Acești tensori sunt similari cu vectorii din librăria NumPy, doar că pot fi utilizați pe un GPU de la Nvidia (CUDA).


## Performanta atinsa 

Setul de date a fost împărțit 80% pentru antrenare și 20% pentru testare. Imaginile au fost redimensionate astfel încat toate să aibă dimensiunea de 224 x 224 x 3, aceași medie și deviație standard. Antrenarea a fost făcută pe un caculator cu placă video Nvidia GeForce GTX 960M timp de 50 de epoci.

### Experiment 1 

- learningRate = 0.001
- optimizator: Adam
- batch size = 128
- cu scheduler care v-a modifica rata de învățare cu 0.1 o dată la 10 epoci

Acuratețea și Pierderea la Antrenare:
![[AccLoss.png]]
#### Testare
Acuratetea la test este 46.2347%

Din matricea de confuzie se poate observa că cele mai greu de identificat sunt tablourile in stilurile mai "noi", cum ar fi cubismul, expresionismul, stilul naiv, iar cel mai usor de identificat este stilul byzantin:
![[Pasted image 20221218114824.png]]
![[Pasted image 20221218115055.png]]
### Experiment 2
- learningRate = 0.01
- optimizator: Adam
- batch size = 128
- cu scheduler care v-a modifica rata de învățare de zece ori o dată la 20 epoci
![[Pasted image 20221218012318.png]]
#### Testare
Acuratetea la test este 29%

### Experiment 3 (mobileNet_PandoraTrain7)
 
- learningRate = 0.01
- optimizator: Adam
- batch size = 64
- cu scheduler care v-a modifica rata de învățare de zece ori o dată la 10 epoci



### Romanian Painting retrieval 

**Pasi efectuați:**
1. Extragere de trăsături
	- Folosim baza de date PandoraRom 
			- ea contine 3371 de poza de la 80 de artiști români
			- fiecare poză are atribuit un stil artistic ( 48 de stiluri artistice în total)
	- Am ales 400 de imagini - cate 5 imagini de la fiecare autor
	- Cele 400 de imagini au fost folosite ca intrare pentru rețeaua MobileNet antrenată anterior
	- Valorile de pe ultimul strat au fost reținute și salvate într-un document excel împreună cu titlul și stilul imaginii
2. Returnarea unei imagini
	- pentru testarea sistemului alege o imagine din baza de date Pandora18
	- se calculează trăsăturile imaginii (valorile de pe ultimul strat)
	- se calculează distanța dintre vectorul de trăsături rezultat și fiecare dintre cele 400 de vectori de trăsături reținuți
	- se returnează primele 5 cele mai mici distanțe și se afișează pozele aferente

Pentru a îmbunătății calitatea sistemului am încercat atât distanța euclidiană, cât și distanța cosinus și, de asemenea, am înmulțit fiecare vector x 20. 

Rezultate obținute: 

![[Rom_Impress.png]]
![[Pasted image 20221215160848.png]]
![[Pasted image 20221215161003.png]]
![[Pasted image 20221215161939.png]]
![[Pasted image 20221215162137.png]]
![[Romantic.png]]




## Comparatie cu rezultate din literatura identificate prin cautare dupa articolul care introduce baza de date


În cadrul lucrării "*Artistic movement recognition by boosted fusion of color structure and topographic description*"[1] s-a introdus baza de date Pandora18 și s-au comparat diverse metode pentru recunoașterea curentului artistic. 

Rezultatele obținute cu rețele neuronale adânci sunt următoarele:
![[tabel4_florea.png]]

RR pentru rețeaua MobileNet antrenată în cadrul acestui proiect a fost 46,23%

Matricea de confuzie prezentată în articol:
![[Pasted image 20221218120944.png]]

Matricea de confuzie obținută (cu varianta cea mai buna a retelei):
![[Pasted image 20221218121451.png]]
Deși nu am putut întrece acuratețea obținută cu AlexNet, soluția prezentată se apropie destul de mult de performanța aflată pe numărul 2.

# Bibliografie

1. Florea, Corneliu, Cosmin Toca, and Fabian Gieseke. "Artistic movement recognition by boosted fusion of color structure and topographic description." _2017 IEEE Winter Conference on Applications of Computer Vision (WACV)_. IEEE, 2017.

2. https://medium.com/analytics-vidhya/image-classification-with-mobilenet-cc6fbb2cd470

3. https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c
