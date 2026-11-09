# CLAUDE.md — lapscale

Leggi questo file per intero a inizio sessione, poi `docs/roadmap.md` e `docs/decisions.md`.

## Cosa è questo repo

Profondità metrica per **laparoscopia monoculare**, solo da immagine. Lo shaft dello strumento (cilindro
di diametro noto) fa da righello e trasforma la depth di una rete monoculare (MoGe) in millimetri.
Metodo: Wei et al., MICCAI 2024, *Enhanced Scale-aware Depth Estimation for Monocular Endoscopic Scenes
with Geometric Modeling* (MonoEndoDepth, github med-air/MonoEndoDepth), con MoGe al posto di Monodepth2.

## Contesto più ampio (non è codice di questo repo)

Dottorato di Leonardo Passoni (PoliMi, NEARLab, supervisore Prof. De Momi) su un sistema di realtà
aumentata per la linfadenectomia nell'esofagectomia Ivor Lewis, con l'IEO. Si proietta sulla vista
endoscopica la posizione dei linfonodi segnata sulla TAC preoperatoria. Il lavoro adatta LyNAR
(Cruciani et al., RA-L 2026, prostatectomia robotica).

- Fase target: **addominale laparoscopica, endoscopio MONOCULARE**, ancora anatomica = tripode celiaco
  (arterie splenica, epatica comune, gastrica sinistra) + pancreas.
- Questo repo copre il lato intraoperatorio: ottenere geometria metrica dal video monoculare, per poi
  registrare il modello TAC. Segmentazione TAC, annotazione, registrazione NON sono qui.
- Il laparoscopio è tenuto da una specializzanda: movimenti, tremori, estrazioni per pulire dal sangue,
  fumo. I frame reali non sono puliti.

## Come funziona il metodo (da non reinventare)

1. **Geometria dello shaft** (stadio `tool`): maschera → due bordi rettilinei → con K e diametro D noto si
   ricava asse 3D e distanza. Diametro apparente ≈ 2·Z·sin(γ/2), con γ angolo tra i due raggi di bordo;
   qui si **inverte**: D noto → Z = D / (2·sin(γ/2)), più una piccola correzione prospettica. Riferimento
   per la formula e il Monte Carlo: `phase0_check.py` di Leonardo (fuori dal repo, vedi sotto).
2. **Scala** (stadio `scale`): sui pixel dello shaft si ha la depth MoGe e la profondità metrica dalla
   geometria; si fitta `mm = a·d + b`. Versione a un solo parametro (`a` = mediana del rapporto) e
   versione affine, **entrambe tenute e confrontate**. Si applica a tutta l'immagine.
3. Leggere la depth MoGe **dentro** la maschera dello shaft, mai sui bordi (discontinuità).

## Stato

- Fatto: scheletro, `camera.py`, stadio 1 `calib` (board / import / moge fallback), wrapper MoGe in
  `depth/`, 19 test sintetici. `lapscale depth|tool|scale|run|eval|viz` NON esistono ancora.
- Prossimo: `tool/geometry.py` (vedi `docs/roadmap.md`). Non richiede reti.

## Decisioni già prese (non rimetterle in discussione senza motivo)

- Stadi separati che comunicano via file, più `run` che li concatena. Backend dietro protocolli.
- MoGe: versione più recente = **MoGe-3** (rilascio 2026-08-18; `Ruicheng/moge-3-vitl` 370M, `-vitg` 1.25B;
  MIT). Offline: MoGe-3. Real time: MoGe-2 vits/vitb in fp16. **MoGe-3 dipende da Triton/FlexGEMM: Linux
  o WSL2, non Windows nativo.** MoGe-2 gira ovunque.
- `fov_x` noto (da calibrazione) va passato a MoGe; se non c'è, lo stima (fallback meno affidabile).
- Nel monoculare **l'errore sulla focale è errore di scala uno a uno**: la calibrazione è prerequisito.
  Il diametro dello strumento idem: va misurato, non assunto.
- Segmentazione dello strumento: la scelta NON è fatta. Candidati: SAM 3, rete semantica con backbone
  chirurgico (es. SurgeNetXL) + fine-tuning, YOLO-seg. Criterio = **precisione dei bordi dello shaft**
  (bias e dispersione in px su ~30-50 frame annotati a mano), NON il Dice. Un bias di 1-2 px per lato
  su uno shaft largo ~110 px vale ~2-4% di scala. Separare shaft e pinza **geometricamente** (RANSAC
  sul contorno), non con classi annotate.
- Il problema dei bordi quasi orizzontali era **solo della stereo** (baseline): nel monoculare non c'è.
- Fine-tuning di MoGe: no, finché la valutazione sulla stereo non mostra che serve.
- La depth non serve a ogni frame: segmentazione ogni frame, depth a 5-10 Hz in parallelo (real time).
- Hardware: **GPU ancora da confermare** (chiedere a Leonardo). Le stime di tempo nei messaggi precedenti
  non sono misurate: il primo benchmark vero è `lapscale bench` (da scrivere).

## Validazione con i dati stereo della collega

Esistono video stereo da Vinci (SX/DX, 1920x1080, 25 fps) con calibrazione `stereo_calib_par_30.xml`.
Lo script `phase0_check.py` (sync, check, refine, measure, selftest) li ha validati: sync a offset 0,
calibrazione par_30 corretta (par_0 no), rifinita con `refine` (dy mediano 1.00 → 0.48 px). Misura dello
shaft: 8.01 ± 0.30 mm contro 8 nominali, **a 42 mm di distanza** (un solo frame utile a 10°).
- Limite noto: validata solo a distanza ravvicinata; il tessuto sta a ~100 mm. Un errore sistematico
  pesa di più lontano.
- Uso previsto in `eval`: video sinistro come input monoculare, stereo come verità, errori per fascia di
  distanza dallo strumento.
- Attenzione al raggio: il paper dichiara 4.5 mm per gli strumenti da Vinci, la misura qui conferma
  8 mm di diametro (raggio 4). Verificare sempre il diametro dello strumento usato.

## Regole di lavoro

- **Prima di dire "fatto": `pytest` e `ruff check src tests` verdi.** Mostra i numeri reali, non quelli
  attesi. Se un test sintetico non torna, indaga e spiegalo (è già successo: la std della focale di
  OpenCV è ottimista fino a ~2x, ora documentato).
- Ogni modulo geometrico ha un **test su dati sintetici con verità nota** (come `tests/synthetic.py`).
- Ogni scelta di metodo va in `docs/decisions.md`; ogni esperimento con risultati in
  `docs/experiment_log.md`; formati dei file in `docs/outputs.md`.
- Stile dell'utente: comunicazione diretta e tecnicamente precisa, **soluzioni minime e oneste** meglio di
  soluzioni rifinite che promettono troppo, **nessuno scope creep**: fai ciò che è chiesto, segnala il
  resto invece di farlo.
- Non inventare riferimenti bibliografici o numeri. Se non sai, dillo. Se una versione di libreria o di un
  modello può essere cambiata, verificala nel codice o nel README ufficiale.
- Dipendenze pesanti (torch, MoGe, SAM) solo come extras opzionali, importate in modo lazy.
- Non committare video, pesi, dati (`.gitignore` già li esclude). Dati sanitari: nulla di identificabile.
- Lingua: **chat in italiano; codice, commenti, docstring e docs in inglese**.

## Comandi

```bash
uv venv && uv pip install -e ".[dev]"     # base (numpy, opencv, pyyaml)
uv pip install -e ".[moge]"               # + torch e MoGe (v3: Linux/WSL2)
pytest                                    # ~20 s
ruff check src tests
lapscale calib --help
```

## Domande aperte (chiedere a Leonardo, non decidere da soli)

- Quale GPU/macchina per sviluppo e per il real time?
- Diametri reali degli strumenti laparoscopici all'IEO (5 o 10 mm?) e ottica usata (0° o 30°)?
- Esiste una calibrazione del laparoscopio IEO? Altrimenti serve un video di scacchiera/ChArUco
  (`docs/calibration.md`).
- Dove sono i video laparoscopici monoculari e che risoluzione/fps hanno?
