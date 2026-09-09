Philosophy
==========

Why pyml exists
---------------

I have studied the exact sciences since childhood — calculus,
probability and statistics, linear algebra among them. My goal in
building pyml was to take these subjects out of the page: derivations
worked by hand, matrix operations performed on paper, and see them
hold up in a real, working implementation, presented in code with the
same rigor and standard practice as the mathematics itself.

Why NumPy and SciPy, and not pure Python
----------------------------------------

Python loops are slow relative to statically typed languages. A model
written in C++ would run faster, but writing machine learning models
directly in C++ is slow and demands careful, often complex, memory
management. The more practical path was to keep Python's readability
while handing the underlying mathematical operations off to C++ —
which is exactly what NumPy's vectorization and SciPy provide.

That said, not every ``for`` loop can be eliminated. In the linear
models, gradient descent's step-by-step decrease in loss is driven by
a plain Python loop over ``max_iter``, with every mathematical
operation inside that loop resting on NumPy.

What building this taught me
----------------------------

Discipline and tooling
~~~~~~~~~~~~~~~~~~~~~~

I had only read about tools like mypy and ruff, and assumed they
belonged to enterprise, production-grade codebases — not to something
as small as a student project. Building pydsa and then pyml changed
that. As a project grows, so does the likelihood of human error, and
tools that enforce consistent design and strict typing turn out to
matter just as much for an educational project as for a commercial
one.

Strict rules cannot always be followed to the letter, though. Machine
learning has its own unwritten conventions — ``X`` in uppercase for a
matrix, ``y`` in lowercase for a vector — and renaming them to
something a general style checker would prefer is itself a mistake:
this is the notation the field and its textbooks actually use.
Knowing where to enforce a rule strictly and where to grant a
deliberate exception turned out to be its own skill.

Strict typing surfaced a related lesson. Enforcing it exposed real
constraints on how the models could be used, and working through
those constraints — rather than loosening the types to avoid them —
was itself a meaningful part of what building pyml taught me.

Numerical stability
~~~~~~~~~~~~~~~~~~~

Writing the models and their tests exposed a problem that never
appears on paper: gradients, distances, and other arithmetic hit real
numerical limits. Floating-point numbers have a finite range, and
formulas that look perfectly stable on a page can diverge or produce
nonsense once actually computed. Guarding against this — clipping
values to a safe range, or standardizing inputs before fitting — turns
out to matter as much for the model's arithmetic as for a human
reader trying to make sense of the output.

The clearest case of this was in division: without guarding the
denominator, the division itself could produce ``NaN`` or ``inf`` in
NumPy. The fix was to add a small constant such as ``1e-10`` to the
denominator before dividing. This was not a default habit but a
deliberate choice, applied only where the underlying computation
could otherwise produce those exact failure modes.

Two years in the making
-----------------------

That pyml came together over a span of a few months is not a
reflection of the mathematics being easy. It reflects the two years
that preceded it. The project itself was written in my third year of
university, but the two years before that were spent in the
university library, working through calculus, probability and
statistics, linear algebra, and programming tools, along with
textbooks by international authors that friends studying abroad
recommended and sent along.

That learning has not stopped. Knowledge in these fields has no fixed
boundary, and I would not claim to have fully mastered any of them —
their depth exceeds what any single book, scoped to its own purpose,
can cover. What drives the learning is not the university calendar or
my year of study, but my own interest in continually updating and
deepening what I know.

The examples themselves went through the same process: early
versions used scikit-learn to generate synthetic datasets, until it
became clear that even that indirect dependency didn't belong in a
project built to avoid exactly this kind of shortcut. Every dataset
in the documentation is now generated with NumPy alone.

Acknowledgments
---------------

This project was not built alone. I am its only contributor, but a
number of people I hold in high regard have shaped the work
indirectly, and their contribution deserves to be named.

Foremost among them is the faculty of the Software Engineering and
Artificial Intelligence department, and the Mathematics department,
within the Faculty of Applied Mathematics and Intellectual
Technologies at the `National University of Uzbekistan <https://nuu.uz/en/>`_.
Their lectures and practicals gave me the room to grow in depth, and their
habit of pointing me toward further reading, and giving their own time outside
of class to my independent study, shaped a lasting respect for them.

I am also indirectly indebted to my brother, Maruf Juraev, who holds a PhD and
currently serves as an Associate Professor in the Department of Cybersecurity.
His precise guidance in the world of software, and the discipline instilled by
his exacting standards, are things I hold in equally high regard.
