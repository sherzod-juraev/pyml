About
=====

.. container:: sd-fs-5

    pyml is a learning project: a machine learning library I built to understand the
    mathematics from the ground up, and to practise real software engineering on something
    that has to work.

*by Sherzod Juraev*

.. only:: html

    .. grid:: 1 1 3 3
        :gutter: 3
        :class-container: sd-mb-4

        .. grid-item-card::
            :text-align: center
            :shadow: sm

            NumPy and SciPy only
            ^^^^^^^^^^^^^^^^^^^^
            No TensorFlow, no PyTorch, just the math.

        .. grid-item-card::
            :text-align: center
            :shadow: sm

            Derived by hand
            ^^^^^^^^^^^^^^^
            Every model starts on paper, then in code.

        .. grid-item-card::
            :text-align: center
            :shadow: sm

            Strictly checked
            ^^^^^^^^^^^^^^^^
            Strict typing, linting and a full test suite.

Why pyml exists
---------------

I have studied the exact sciences since childhood: calculus, probability and statistics, linear
algebra. I wanted to take these subjects off the page, so in pyml I work the derivations out by
hand, do the matrix operations on paper, and then check that they hold up in a real
implementation. The code is meant to be held to the same standard as the mathematics.

It came together over a few months, but that is only the last part. The project was written in my
third year of university, after two years in the university library, working through calculus,
probability and statistics, linear algebra and programming tools. Some of the textbooks were by
international authors, recommended and sent to me by friends studying abroad.

The learning has not stopped. These fields have no fixed boundary, and I would not claim to have
mastered any of them. What keeps me going is not the university calendar but my own interest in
updating and deepening what I know.

How it is built
---------------

Every model goes through the same steps. I work out the mathematics on paper first, then decide
the architecture, then write the implementation myself. After that come the tests, the docstrings
and the documentation, and a set of checks that must pass before a release: strict typing with
mypy, linting with ruff, the test suite, and a documentation build that treats warnings as errors.

Python loops are slow compared with compiled languages. A model written in C++ would run faster,
but writing machine learning models directly in C++ takes longer and needs careful memory
management. The practical path was to keep Python's readability and hand the heavy mathematical
operations to compiled code, which is what NumPy's vectorization and SciPy provide.

Not every ``for`` loop can be removed, though. In the linear models, gradient descent runs as a
plain Python loop over ``max_iter``, and every operation inside that loop is NumPy.

How the examples are made
-------------------------

The examples are plain Python scripts. They run when the documentation is built, on real
datasets from `pyml-datasets <https://github.com/sherzod-juraev/pyml-datasets>`__, a companion
package of my own. The numbers, the plots and the description of the dataset on a page come from
the code and not from me typing them, so an example cannot disagree with the library. The small
plots in the API reference are the exception: they use generated data, because they only show
the shape of an idea.

What building it taught me
--------------------------

Discipline and tooling
~~~~~~~~~~~~~~~~~~~~~~

I had only read about tools like mypy and ruff, and assumed they belonged to big production
codebases, not to a student project. Building pydsa and then pyml changed that. As a project
grows, so does the chance of a human mistake, and tools that enforce a consistent design and
strict typing matter just as much for an educational project as for a commercial one.

Strict rules cannot always be followed to the letter, though. Machine learning has its own
conventions, such as ``X`` in uppercase for a matrix and ``y`` in lowercase for a vector. Renaming
them to what a general style checker prefers would be a mistake, because this is the notation the
field and its textbooks use. Knowing where to enforce a rule and where to allow a deliberate
exception turned out to be a skill of its own.

Strict typing taught me a related lesson. It exposed real constraints on how the models can be
used, and working through those constraints, instead of loosening the types to avoid them, was a
real part of what building pyml taught me.

Numerical stability
~~~~~~~~~~~~~~~~~~~

Writing the models and their tests showed me a problem that does not appear on paper: gradients,
distances and other arithmetic hit real numerical limits. Floating-point numbers have a finite
range, and a formula that looks stable on the page can diverge or give nonsense once it is
computed. Guarding against that, by clipping values to a safe range or by standardizing the inputs
before fitting, matters as much as the formula itself.

The clearest case was division. Without a guard on the denominator, the division can produce
``NaN`` or ``inf`` in NumPy. The fix was to add a small constant such as ``1e-10`` to the
denominator before dividing. I did not do this by habit, only where the computation could
otherwise fail in exactly that way.

What pyml is, and is not
------------------------

pyml is a learning project, not a replacement for scikit-learn. It is not published on PyPI, the
linear models use plain gradient descent, so they are slower and need standardized features, and
speed has not been the goal. What I aimed for is code that is readable, correct and checked, so
that a reader can follow a model from the formula to the code.

To try it, start with :doc:`getting-started`, then look through the :doc:`examples/index`.

About the author
----------------

.. only:: html

    .. image:: _static/branding/author-sj.jpg
        :alt: SJ, the monogram of Sherzod Juraev
        :align: left
        :class: author-photo sd-rounded-circle sd-mr-3

I am a student at the |nuu_link|_, in the Faculty of Applied Mathematics and Intellectual
Technologies. I have been drawn to the exact sciences since childhood, and I learn them by
building: pyml sits next to two other projects on GitHub, `pydsa
<https://github.com/sherzod-juraev/pydsa>`__ (data structures and algorithms) and
`pyml-datasets <https://github.com/sherzod-juraev/pyml-datasets>`__ (the datasets used in
pyml's examples). I write the implementation myself, and I care about the details that make
code trustworthy: tests, types and documentation.

.. only:: html

    .. grid:: auto
        :gutter: 2
        :class-container: sd-mt-3

        .. grid-item::
            :columns: auto

            .. button-link:: https://github.com/sherzod-juraev
                :color: primary
                :outline:
                :tooltip: GitHub
                :class: sd-rounded-pill sd-px-4 sd-py-2

                :fab:`github;sd-fs-4`

        .. grid-item::
            :columns: auto

            .. button-link:: https://www.linkedin.com/in/sherzod-juraev
                :color: primary
                :outline:
                :tooltip: LinkedIn
                :class: sd-rounded-pill sd-px-4 sd-py-2

                :fab:`linkedin-in;sd-fs-4`

        .. grid-item::
            :columns: auto

            .. button-link:: https://x.com/SJuraev74508
                :color: primary
                :outline:
                :tooltip: X
                :class: sd-rounded-pill sd-px-4 sd-py-2

                :fab:`x-twitter;sd-fs-4`
