# requirements

see [requirements.txt](./src/requirements.txt)

# build the package

I not really sure if it's necessary.

# about data

It will automatically download all the files it needs.

```shell
python -m pip install -e ./
```
# rule-based segmentator evaluation

## generate combined prescription audio data

```shell
python -m acpsr generate
```

## evualation

```shell
python -m acpsr evaluation
```