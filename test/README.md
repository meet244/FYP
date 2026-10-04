# ClassScribe test fixtures

Synthetic **ITC604 — Introduction to Neural Networks** corpus for local multimodal
tests. Nothing here is a real university file. Text in the PDFs is extractable
(`pypdf`), so RAG does not depend on OCR.

## Layout

| Path | Use in the app |
|---|---|
| `syllabus/ITC604-Neural-Networks-Syllabus.pdf` | Subject → Files → syllabus upload |
| `slides/L01-Perceptrons-and-MLPs.pdf` | Files → materials (PDF) |
| `readings/perceptron-learning-handout.pdf` | Files → materials (PDF) |
| `readings/activation-functions.md` | Files → materials (markdown) |
| `readings/xor-lab-notes.txt` | Files → materials (text) |
| `media/perceptron-diagram.png` | Files → materials (image) |
| `media/mlp-architecture.png` | Files → materials (image) |
| `media/xor-decision-regions.png` | Files → materials (image) |
| `voice/L01-perceptrons-en.mp3` | Files → lecture audio (~96 s, Indian English, some Hindi) |
| `voice/L01-perceptrons-hi.mp3` | Optional second lecture (~25 s, Hindi) |
| `voice/*.wav` | Same clips as 16 kHz mono PCM (pipeline native) |
| `voice/*-script.txt` | Reference wording for the TTS clips |

Rebuild PDFs (does not regenerate audio or images):

```bash
backend/.venv/bin/python test/_build_fixtures.py
```

```bash
# API must be running
CLASSSCRIBE_API=http://127.0.0.1:8002 backend/.venv/bin/python test/load.py
```

## Questions that hit different modalities

- What is a perceptron? *(slides + voice + syllabus unit I)*
- Why can a single perceptron not learn XOR? *(handout section 2, xor image, lab notes)*
- Give weights for a two-layer XOR network. *(handout section 3)*
- How many parameters does a 4-5-4-2 MLP have? *(syllabus unit II, handout section 5)*
- When do you use ReLU vs sigmoid? *(activation-functions.md)*
- State the perceptron convergence theorem. *(syllabus unit I, lecture audio)*
- Give me five practice questions on linear separability. *(quiz route)*

WAV/MP3 are TTS (`say` voices Rishi and Lekha), not a classroom mic. Good for
pipeline plumbing; not a WER number you can put in the paper.
