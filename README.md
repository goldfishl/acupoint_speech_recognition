# build the package

I not really sure if it's necessary.

```shell
python -m pip install -e ./
```

# about data

It will automatically download all the files it needs from my server.

# rule-based segmentator evaluation

generate combined prescription audio data

```shell
python -m acpsr generate
```

evualation

```shell
python -m acpsr evaluation > ./result/rule_based_eval.txt && cat ./result/rule_based_eval.txt
```