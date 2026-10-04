"""Build extractable PDFs for ClassScribe multimodal tests. Run from repo root:

    backend/.venv/bin/python test/_build_fixtures.py
"""
from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent


class Doc(FPDF):
    title_text = ""
    subtitle = ""

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, self.title_text, align="L")
        self.ln(12)
        self.set_text_color(20, 20, 20)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"ClassScribe test fixture  -  {self.page_no()}/{{nb}}", align="C")
        self.set_text_color(20, 20, 20)

    def banner(self) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 11)
        self.multi_cell(0, 7, "ITC604  -  Introduction to Neural Networks")
        self.ln(2)
        self.set_x(self.l_margin)
        self.set_font("Times", "B", 22)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 9, self.title_text)
        self.ln(2)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 6, self.subtitle)
        self.set_text_color(20, 20, 20)
        self.ln(4)

    def heading(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.ln(3)
        self.set_font("Helvetica", "B", 13)
        self.multi_cell(0, 7, text)
        self.ln(1)

    def p(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Times", "", 11)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def bullets(self, items: list[str]) -> None:
        self.set_font("Times", "", 11)
        for item in items:
            self.set_x(self.l_margin)
            self.multi_cell(0, 6, f"  -  {item}")
        self.ln(2)

    def unit(self, code: str, title: str, hours: str, prose: str, outcomes: list[str]) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 12)
        self.multi_cell(0, 7, f"Unit {code}: {title}  ({hours})")
        self.set_x(self.l_margin)
        self.set_font("Times", "I", 11)
        self.multi_cell(0, 6, prose)
        self.ln(1)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 10)
        self.multi_cell(0, 6, "Learning outcomes")
        self.bullets(outcomes)


def write_pdf(path: Path, title: str, subtitle: str, build) -> None:
    pdf = Doc()
    pdf.title_text = title
    pdf.subtitle = subtitle
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.banner()
    build(pdf)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    print("wrote", path.relative_to(ROOT.parent))


def syllabus(pdf: Doc) -> None:
    pdf.p(
        "Department of Information Technology. This syllabus is a ClassScribe test "
        "fixture: every unit is written as lecture narration, not a keyword list, so "
        "syllabus ingestion has real prose to rewrite. Teaching hours assume 12 weeks, "
        "3 hours of lecture plus 2 hours of lab."
    )
    pdf.heading("Course details")
    pdf.bullets(
        [
            "Code: ITC604    Credits: 4    Level: undergraduate, fifth semester",
            "Prerequisites: linear algebra (vectors, matrices), basic probability, Python",
            "Assessment: 20% labs, 30% midterm, 50% end-semester examination",
            "Text: Goodfellow, Bengio, Courville - Deep Learning (selected chapters)",
        ]
    )
    pdf.unit(
        "I",
        "The perceptron and linear classification",
        "8 hours",
        "In this unit we treat the perceptron as a linear binary classifier. Inputs x1 "
        "through xn are multiplied by weights w1 through wn, a bias b is added, and a "
        "step activation decides the class. The decision boundary is the hyperplane "
        "w dot x + b = 0. We prove that a single perceptron can realise AND, OR and "
        "NOT, and we show Rosenblatt's perceptron criterion and the mistake-driven "
        "update w := w + eta * y * x. Students should leave able to implement this "
        "update on a linearly separable toy set and to state the perceptron "
        "convergence theorem in one sentence: if a separating hyperplane exists, the "
        "algorithm finds one in a finite number of updates.",
        [
            "Write the perceptron forward pass and the mistake-driven weight update.",
            "Construct weight vectors for AND, OR and NAND on two binary inputs.",
            "State when the algorithm is guaranteed to halt.",
        ],
    )
    pdf.unit(
        "II",
        "The XOR problem and multilayer networks",
        "8 hours",
        "XOR is the canonical example of a function that is not linearly separable: "
        "the points (0,0) and (1,1) belong to class 0 while (0,1) and (1,0) belong "
        "to class 1, and no straight line separates them. Minsky and Papert used this "
        "to show the limits of a single layer. The fix is a hidden layer. A two-layer "
        "network can compute XOR by forming the intermediate features x1 AND NOT x2 "
        "and NOT x1 AND x2, then OR-ing those features. We introduce the multilayer "
        "perceptron as a composition of affine maps and pointwise activations, and we "
        "count parameters: for layers of size n0, n1, n2 the weight matrices are "
        "n1 by n0 and n2 by n1, plus a bias per hidden and output unit.",
        [
            "Draw the XOR points and explain why one hyperplane is not enough.",
            "Hand-design a two-layer network that realises XOR with step activations.",
            "Count the parameters of a 4-5-4-2 fully connected network.",
        ],
    )
    pdf.unit(
        "III",
        "Backpropagation and gradient descent",
        "10 hours",
        "Training replaces the step function with a differentiable activation so that "
        "a loss L(w) can be differentiated. Mean squared error and binary cross-entropy "
        "are the two losses we use in this course. Backpropagation applies the chain "
        "rule from the output backwards: the gradient with respect to a layer's weights "
        "is the outer product of that layer's input and the incoming error signal. "
        "Stochastic gradient descent updates w := w - eta * grad on a minibatch. We "
        "discuss learning-rate schedules, vanishing gradients in saturating sigmoids, "
        "and why ReLU became the default hidden activation. Students implement "
        "backprop for a two-layer network on XOR and watch the loss fall.",
        [
            "Derive dL/dw for a single linear unit with squared loss.",
            "Explain vanishing gradients for sigmoid activations in deep stacks.",
            "Train a two-layer MLP on XOR and report loss after 200 epochs.",
        ],
    )
    pdf.unit(
        "IV",
        "Generalisation, overfitting and regularisation",
        "6 hours",
        "A network that memorises the training set and fails on new points is said to "
        "overfit. We split data into train, validation and test, plot learning curves, "
        "and stop when validation loss rises. Regularisation methods in this unit are "
        "L2 weight decay, dropout, and data augmentation. We also introduce the bias-"
        "variance trade-off: too small a network underfits, too large a network "
        "overfits unless regularised. This unit closes the lecture course; labs "
        "continue with a small image classifier using the same ideas.",
        [
            "Diagnose overfit vs underfit from a pair of learning curves.",
            "State the effect of increasing L2 weight decay on the effective capacity.",
            "Describe dropout at train time versus test time.",
        ],
    )
    pdf.heading("Laboratory list")
    pdf.bullets(
        [
            "Lab 1: perceptron on a linearly separable 2-D set; plot the moving boundary.",
            "Lab 2: show XOR failure of a single layer; implement a 2-2-1 MLP.",
            "Lab 3: backprop from scratch in NumPy; compare eta in {0.01, 0.1, 1.0}.",
            "Lab 4: regularisation - dropout vs L2 on a noisy polynomial fit.",
        ]
    )


def slides(pdf: Doc) -> None:
    pdf.heading("Learning goals for this lecture")
    pdf.bullets(
        [
            "Write y = step(w . x + b) and name every symbol.",
            "Give weights that realise AND and OR.",
            "Show that XOR is not linearly separable.",
            "Explain, in one paragraph, why a hidden layer fixes XOR.",
        ]
    )
    pdf.heading("The biological sketch, then the model")
    pdf.p(
        "A biological neuron collects incoming spikes on its dendrites, sums them "
        "near the soma, and fires an axon spike if the sum crosses a threshold. The "
        "perceptron is a cartoon of that: real-valued inputs, a weighted sum, a "
        "threshold. It is not a model of cortex. It is a linear classifier that we "
        "can train."
    )
    pdf.heading("Forward pass")
    pdf.p(
        "Let x be an n-dimensional vector. Let w be the same length. The pre-activation "
        "is z = w1 x1 + w2 x2 + ... + wn xn + b. The output is y_hat = 1 if z >= 0, "
        "else 0. Equivalently y_hat = step(z). The set of points with z = 0 is a "
        "hyperplane. Changing w rotates the hyperplane; changing b shifts it."
    )
    pdf.heading("Boolean functions")
    pdf.p(
        "AND: weights (1, 1), bias -1.5. Both inputs must be 1 before z is positive. "
        "OR: weights (1, 1), bias -0.5. Either input is enough. NAND is AND with the "
        "sign of the weights and bias flipped. XOR cannot be written this way. Plot "
        "the four points and try: any line that puts (0,1) and (1,0) on the positive "
        "side puts at least one of (0,0) or (1,1) there too."
    )
    pdf.heading("Perceptron training rule")
    pdf.p(
        "Present examples one at a time. If the prediction matches the label, do "
        "nothing. If the true label is 1 and we predicted 0, add eta * x to w and "
        "eta to b. If the true label is 0 and we predicted 1, subtract. This is "
        "mistake-driven learning. If a separating hyperplane exists, the algorithm "
        "finds one in a finite number of mistakes. That is the perceptron convergence "
        "theorem. If the data is not linearly separable, the updates never settle."
    )
    pdf.heading("From one layer to two")
    pdf.p(
        "Hidden unit h1 computes x1 AND NOT x2. Hidden unit h2 computes NOT x1 AND x2. "
        "The output unit ORs h1 and h2. That network is XOR. In the next lecture we "
        "replace the step with a sigmoid or ReLU so we can take gradients and learn "
        "those hidden features instead of designing them by hand."
    )
    pdf.heading("What to read before Lab 1")
    pdf.p(
        "Implement the perceptron in about twenty lines of Python. Use eta = 0.1. "
        "Draw the line w1 x + w2 y + b = 0 after every epoch. The handout "
        "perceptron-learning-handout.pdf works the same examples with numbers."
    )


def handout(pdf: Doc) -> None:
    pdf.heading("1.  Worked AND example")
    pdf.p(
        "Inputs are (x1, x2) in {0,1}^2. Target t = x1 AND x2. Initialise w = (0, 0), "
        "b = 0, eta = 1. First example (1, 1), t = 1. z = 0, prediction 0, a mistake. "
        "Update: w := (1, 1), b := 1. Second example (1, 0), t = 0. z = 1+0+1 = 2, "
        "prediction 1, a mistake. Update: w := (0, 1), b := 0. Continue cycling the "
        "four points. A solution you should reach: w = (1, 1), b = -1.5 after scaling, "
        "or any positive multiple. Check: only (1,1) yields z >= 0."
    )
    pdf.heading("2.  Why XOR breaks the same procedure")
    pdf.p(
        "Label t(0,0)=0, t(1,1)=0, t(0,1)=1, t(1,0)=1. Suppose a line ax + by + c = 0 "
        "separates them. Then a*0 + b*0 + c < 0 so c < 0; a + b + c < 0; b + c > 0; "
        "a + c > 0. Adding the last two inequalities: a + b + 2c > 0. But a + b + c < 0 "
        "and c < 0 imply a + b + 2c < 0. Contradiction. No such line exists, so the "
        "perceptron updates cycle forever. This is the numerical form of the picture "
        "in xor-decision-regions.png."
    )
    pdf.heading("3.  Two-layer XOR by hand")
    pdf.p(
        "Hidden 1: w = (1, -1), b = -0.5  ->  fires on (1,0). "
        "Hidden 2: w = (-1, 1), b = -0.5  ->  fires on (0,1). "
        "Output: w = (1, 1), b = -0.5  ->  OR of the two hidden bits. "
        "Forward (1,0): h = (1, 0), y = 1. Forward (1,1): h = (0, 0), y = 0. "
        "Forward (0,0): h = (0, 0), y = 0. Forward (0,1): h = (0, 1), y = 1."
    )
    pdf.heading("4.  Gradient for a linear unit")
    pdf.p(
        "Let y_hat = w . x + b and L = 0.5 (y_hat - t)^2. Then dL/dw = (y_hat - t) x "
        "and dL/db = (y_hat - t). One SGD step: w := w - eta (y_hat - t) x. Compare "
        "this to the perceptron rule: both move the weights in the direction of x "
        "when the prediction is too small, but SGD uses a continuous error instead of "
        "a hard mistake. With a sigmoid, replace (y_hat - t) by (y_hat - t) * y_hat * "
        "(1 - y_hat) if L is squared error, or simply (y_hat - t) if L is cross-entropy."
    )
    pdf.heading("5.  Numbers students always mix up")
    pdf.p(
        "Learning rate eta is not a probability. Typical starting values: 0.1 for "
        "perceptron on toy data, 0.01 for SGD on a sigmoid MLP, 0.001 for Adam on a "
        "deeper net. Epoch means one pass over the training set. Minibatch 32 is a "
        "default, not a law. Parameter count for 4-5-4-2: 4*5 + 5 + 5*4 + 4 + 4*2 + 2 "
        "= 20 + 5 + 20 + 4 + 8 + 2 = 59. Bias terms are easy to forget and examiners "
        "notice."
    )
    pdf.heading("6.  Glossary for the chat corpus")
    pdf.p(
        "Perceptron: linear binary classifier with a step activation. Hyperplane: the "
        "set w . x + b = 0. Linearly separable: some hyperplane classifies every "
        "training point correctly. Hidden layer: intermediate units whose outputs are "
        "inputs to the next layer. Backpropagation: chain-rule gradients from loss to "
        "every weight. ReLU: max(0, z), the default hidden activation after 2011. "
        "Overfitting: train loss low, test loss high. Dropout: randomly zero hidden "
        "units at train time. These terms appear in the lecture recording and in the "
        "diagram captions, so retrieval should be able to cite all three modalities."
    )


def main() -> None:
    write_pdf(
        ROOT / "syllabus" / "ITC604-Neural-Networks-Syllabus.pdf",
        "Course syllabus",
        "Undergraduate elective  -  12 weeks  -  test fixture for ClassScribe",
        syllabus,
    )
    write_pdf(
        ROOT / "slides" / "L01-Perceptrons-and-MLPs.pdf",
        "Lecture 1 - Perceptrons, linear separation, and XOR",
        "ITC604  -  use with the syllabus units I and II",
        slides,
    )
    write_pdf(
        ROOT / "readings" / "perceptron-learning-handout.pdf",
        "Worked examples: AND, XOR, and the first gradient",
        "Companion to Lecture 1  -  numbers you can check by hand",
        handout,
    )


if __name__ == "__main__":
    main()
