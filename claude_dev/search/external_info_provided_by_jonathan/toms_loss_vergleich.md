# Toms Loss-Vergleich (Praktikumsbericht) und Bezug zu Haozes Setup

## Wie Tom den Loss berechnet

**Finale Wahl: NLL (Negative Log-Likelihood, Abschnitt 4).** Das Modell gibt per Log-Softmax eine Wahrscheinlichkeitsverteilung über eine 64×64-Heatmap aus, also über 4096 Pixel. Der Loss ist die negative logarithmierte Wahrscheinlichkeit am Pixel der echten nächsten Fixation:

$$\text{Loss} = -\log \hat{p}(k^*)$$

gemittelt über alle Vorhersageschritte im Batch. Das ist normale Cross-Entropy über 4096 „Klassen“ mit One-Hot-Target (1 am richtigen Pixel, 0 überall sonst), ohne jedes Glätten. Liegt viel Wahrscheinlichkeit auf dem richtigen Pixel, ist der Loss klein; liegt fast nichts drauf, ist er groß.

**Ausgangspunkt: KL gegen ein Gauß-Target.** Das ist dieselbe Idee wie Haozes Default: KL-Divergenz gegen eine Gauß-Heatmap, die auf der echten Fixation zentriert ist, mit σ in Heatmap-Pixeln. Das Modell wird dabei auch dafür belohnt, Wahrscheinlichkeit auf Nachbarpixel zu legen. Da das Target fest ist, entspricht KL bis auf eine Konstante der Cross-Entropy mit weichem Target. NLL ist also einfach der Grenzfall σ → 0, ganz ohne Verschmieren.

## Was er herausgefunden hat (Tabelle 2, 50k Bilder, sonst identisch)

| Objective | NSS | LL (bits) | AUC |
|---|---|---|---|
| KL, σ=4 | 2.074 | 1.602 | 0.856 |
| KL, σ=2 | 2.332 | 1.803 | 0.865 |
| KL, σ=1 | 2.404 | 1.844 | 0.867 |
| NLL | 2.413 | 1.853 | 0.867 |

Fast der ganze Gewinn kommt davon, das Gauß-Target schmaler zu machen: σ=4 → 2 bringt +0.258 NSS, σ=1 → NLL nur noch +0.009. Die eigentliche Erkenntnis ist „schärfer ist besser“; dass NLL besser ist als KL, gilt nur knapp.

## Bezug zu Haozes Setup

- **Einordnung von sd=2.5:** Auf demselben 64×64-Grid liegt das zwischen Toms σ=2 und σ=4, also in dem Bereich, wo schärfer bei Tom noch deutlich geholfen hat. Relativ zur Bildseite sind das 3,9 %, gegenüber 3,1 % bei Toms σ=2.
- **Warum das nicht direkt übertragbar sein muss:**
  - Tom misst alles exakt am Ground-Truth-Pixel (NSS, LL, AUC). Das begünstigt automatisch scharfe Targets, was er selbst anmerkt. Wenn bei Haoze z. B. zählt, ob man in der Box landet, kann ein breiteres Target genauso gut oder sogar besser sein.
  - Die Aufgaben unterscheiden sich: Tom sagt stochastische nächste Fixationen bei jedem Schritt vorher, Haoze einen einzelnen Box-Mittelpunkt, und das nur auf dem letzten Glimpse supervisiert.
  - Jede Konfiguration bei Tom ist ein einzelner Run, nur bei einer Datenmenge getestet, mit nur σ ∈ {1, 2, 4}.
