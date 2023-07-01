# Loading from CSV

Scallop can be used along with existing datasets loaded from CSVs.
This is usually achieved with annotating on specific relations.
For example, assuming we have a file `edge.csv`,

``` csv
0,1
1,2
```

we can load the content of it into a relation `edge` in Scallop using the following syntax

``` scl
@file("edge.csv")
type edge(from: usize, to: usize)

rel path(a, c) = edge(a, c) or path(a, b) and edge(b, c)

query path
```

In particular, we annotate the `@file(...)` attribute onto the relation type declaration `type edge(...)`.
The file name is written inside the `@file` attribute.
We require the relation to be declared with types in order for it to be loaded with CSV file content.
Depending on the type declaration, the file content will be parsed into values of certain types.

From here, the `edge` relation will be loaded with the content `(0, 1)` and `(1, 2)`.
After executing the Scallop program above, we would obtain the result `path` being `(0, 1)`, `(0, 2)`, and `(1, 2)`.

Certainly, there are many ways to load CSV.
In this section, we introduce the various ways to configure the CSV loading.

## Headers

There are CSV files with headers.
Suppose we have the following CSV file

``` csv
from,to
0,1
1,2
```

To load this file, we would need to add an additional argument `header=true` to the `@file` attribute:

``` scl
@file("edge.csv", header=true)
type edge(from: usize, to: usize)
```

Note that by default we assume that CSV files don't have headers.

## Deliminators

By default, we assume the values inside of the CSV file are deliminated by commas `','`.
In case where CSV files have values deliminated by other characters, such as tabs `'\t'`, we would need to specify that in the `@file` attribute:

``` scl
@file("edge.csv", deliminator="\t")
type edge(from: usize, to: usize)
```

Note that deliminators cannot be of multiple characters.

## Parsing Field-Value Pairs

There are many CSV tables which have a lot of columns.
One way is to specify all the fields and their types, like the following.

``` scl
type table(field1: type1, field2: type2, ..., fieldn: typen)
```

However, this might be very hard to encode.
Therefore, we provide another way of parsing CSV files into relations, by using primary keys and field-value pairs.
Let's assume we have the following CSV file:

``` csv
student_id,name,year,gender
0001,alice,2020,female
0002,bob,2021,male
```

We see that `student_id` can serve as the primary key of this table.
With this, it can be loaded into the following relation

``` scl
@file("student.csv", keys="student_id")
type table(student_id: usize, field: String, value: String)
```

By specifying `keys="student"`, we tell Scallop that `student_id` should be viewed as unique primary keys.
The rest of the two elements are `field` and `value`, both need to be typed `String`s.
As a result, it produces the following 6 facts in the `table` relation:

```
(1, "name", "alice"), (1, "year", "2020"), (1, "gender", "female"),
(2, "name", "bob"),   (2, "year", "2021"), (2, "gender", "male")
```

Note that there could be more than one keys.
Consider the following table

```
student_id,course_id,enroll_time,grade
0001,cse100,fa2020,a
0001,cse101,sp2021,a
0002,cse120,sp2021,b
```

We see that the combination of `student_id` and `course_id` form the unique primary keys.
In this case, they can be loaded using the following syntax:

``` scl
@file("enrollment.csv", keys=["student_id", "course_id"])
type enrollment(student_id: usize, course_id: String, field: String, value: String)
```

By setting `keys` to be a list `["student_id", "course_id"]`, the `student_id` field is the first primary key and `course_id` is the second.
There are still two additional arguments for the `enrollment` relation.
In general, the arity of the relation will be the number of primary keys plus 2.

## Specifying Fields to Load

In case not all fields are desired when loading, one can use the `fields` argument to specify what to load.
Consider the same enrollment table encoded in CSV:

``` csv
student_id,course_id,enroll_time,grade
0001,cse100,fa2020,a
0001,cse101,sp2021,a
0002,cse120,sp2021,b
```

If we only want to get everything but omit the `enroll_time` column, we can do

``` scl
@file("enrollment.csv", fields=["student_id", "course_id", "grade"])
type enrollment(student_id: usize, course_id: String, grade: String)
```

This can also work in conjunction with the `keys` argument.
In this case, we do not need to specify the primary keys.

``` scl
@file("enrollment.csv", keys=["student_id", "course_id"], fields=["grade"])
type enrollment(student_id: usize, course_id: String, field: String, value: String)
// The following facts will be obtained
//   enrollment(1, "cse100", "grade", "a")
//   enrollment(1, "cse101", "grade", "a")
//   enrollment(2, "cse120", "grade", "b")
```
