# Tags and Provenance

Scallop's probabilistic semantics is realized by the *Provenance Semiring* framework.
Inside of this framework, each fact can be *tagged* by an extra piece of information, which we call *tag*.
Such information is propagated throughout the execution of Scallop program according to the *provenance*, which is the mathematical object defining how tags propagate.

## Motivating Probabilistic Example

The following example shows a fact `earthquake()` being tagged by a probability `0.03` (earthquake could happen with a 0.03 probability):

``` scl
rel 0.03::earthquake()
```

Concretely, we have an *(external) tag space* of \\([0, 1]\\), which contains real numbers between 0 and 1, which is the space of probabilities.
Similarly, we define another tagged fact `burglary()`:

``` scl
rel 0.20::burglary()
```

We can declare a rule saying that, "when earthquake or burglary happens, an alarm will go off".

``` scl
rel alarm() = earthquake() or burglary()
query alarm
```

Remember that the facts `earthquake()` and `burglary()` are probabilistic.
Intuitively, the derived fact `alarm()` will also be associated with a derived probability.
Based on probability theory, we have

\\[
\begin{align}
\Pr(\text{alarm})
&= \Pr(\text{earthquake} \vee \text{burglary}) \\\\
&= 1 - \Pr(\neg \text{earthquake} \wedge \neg \text{burglary}) \\\\
&= 1 - \Pr(\neg \text{earthquake}) \cdot \Pr(\neg \text{burglary}) \\\\
&= 1 - (1 - \Pr(\text{earthquake})) \cdot (1 - \Pr(\text{burglary})) \\\\
&= 1 - (1 - 0.03) (1 - 0.20) \\\\
&= 1 - 0.97 \times 0.8 \\\\
&= 0.224
\end{align}
\\]

This is indeed what we get if we use the `topkproofs` provenance (which we discuss later in the chapter) with the `scli` Scallop interpreter:

```
> scli alarm.scl
alarm: {0.224::()}
```
