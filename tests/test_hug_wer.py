from evaluate import load
wer = load("wer")

predictions = ["this is the prediction", "there is an other sample"]
references = ["this is the reference", "there is another one"]
wer_score = wer.compute(predictions=predictions, references=references)
print(wer_score)
0.5


predictions = ["Yn ddiweddar, roedd y nomenclatrwydd wedi'i ddatblygu i'r cyfansoddiadau sy'n cael eu defnyddio heddiw."]
references = ["Eventually the nomenclature was standardized to the conventions used today."]
wer_score = wer.compute(predictions=predictions, references=references)
print(wer_score)
