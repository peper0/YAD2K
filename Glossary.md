komórka - jeden "piksel" wyjściowej warsty sieci konwolucyjnej, odpowiata blokowi 32x32 piksele obrazu wejściowego
pw, ph - szerokość i wysokość w pikselach
aw, ah - wymiary w stosunku do anchora; aw=pw/anchor.pw
tw, th - tw=log(aw); aw=exp(pw)
px, py - położenie pikselowe wzglęgem lewego górnego rogu
ix, iy - położenie jako frakcja odpowiedniego wymiaru obrazu, tj. współrzędne w zakresie [0,1); ix=px/image.pw
dx, dy - przesunięcie względem lewego górnego rogu komórki
tx, ty - simgoid(tx)=dx
