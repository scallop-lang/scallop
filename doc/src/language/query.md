# Writing Queries

Consider the following example of classes, students, and enrollments, and that we want to compute the number of students who have enrolled in at least one CS class.

``` scl
// There are three classes
rel classes = {0, 1, 2}

// Each student is enrolled in a course (Math or CS)
rel enroll = {
  ("tom", "CS"), ("jenny", "Math"), // Class 0
  ("alice", "CS"), ("bob", "CS"), // Class 1
  ("jerry", "Math"), ("john", "Math"), // Class 2
}

// Count how many student enrolls in CS course
rel num_enroll_cs(n) = n := count(s: enroll(s, "CS"))
```

Normally, executing a program would result in `scli` outputting every single relation.

```
classes: {(0), (1), (2)}
num_enroll_cs: {(3)}
enroll: {("alice", "CS"), ("bob", "CS"), ("jenny", "Math"), ...}
```

However, we might only be interested in the relation named `num_enroll_cs`.
In this case, we write a *query* using the `query` keyword:

``` scl
query num_enroll_cs
```

In this case, only the relation `num_enroll_cs` will be output:

```
num_enroll_cs: {(3)}
```

## Atomic Query

One can also write atomic query if we just want to get a part of the relation.
For instance, consider the fibonacci example:

``` scl
type fib(x: i32, y: i32)
rel fib = {(0, 1), (1, 1)}
rel fib(x, y1 + y2) = fib(x - 1, y1) and fib(x - 2, y2) and x <= 10
query fib(8, y) // fib(8, y): {(8, 34)}
```

In this case, we are just looking at the 8-th fibonacci number, which is 34.
