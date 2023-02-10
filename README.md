# build the package

I not really sure if it's necessary.

```shell
python -m pip install -e ./
```

# about data

It will automatically download all the files it needs.

# rule-based segmentator evaluation

## generate combined prescription audio data

```shell
python -m acpsr generate
```

## evualation

```shell
python -m acpsr evaluation
```