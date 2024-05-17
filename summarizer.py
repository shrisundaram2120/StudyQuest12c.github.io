import gradio as gr
from gradio.mix import Parallel, Series
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSeq2SeqLM

title = "Text Summarizer"
description = "Past an article text or other text. Submit the text and the machine will create four summaries based on words in the text. Which sentences in the text are the most important for the summaries? Which summaries are better for your case?"
examples = [

    ["""""""""
  Hong Kong health authorities on Wednesday began a city-wide search for the contacts of a Covid-19 patient from a suspected dance cluster and ordered a Royal Caribbean "cruise to nowhere" ship with 3,700 people onboard to return to port early. 

The latest hunt was sparked by a 62-year-old woman who danced with some 20 friends at Victoria Park and the Causeway Bay Community Centre on New Year's Eve. Two of the fellow dancers, one of whom was a domestic helper, came up positive in preliminary tests.

The 62-year-old was said to have contracted the virus from her 28-year-old flight attendant daughter, who returned to Hong Kong on December 27 and had onset of symptoms on December 29.

It was only on January 1 that the 62-year-old was classified as a close contact and being brought to a quarantine facility.

The helper's employer and eight other of her close contacts then went on a "cruise to nowhere" journey on January 2, which was due to return on January 6. 

As part of its coronavirus restrictions, Hong Kong has restricted cruises to short trips in nearby waters, with ships asked to operate at reduced capacity and to only allow vaccinated passengers who test negative for the virus. 

The "Spectrum of the Seas" ship had about 2,500 passengers and 1,200 staff on board. The nine close contact passengers were isolated from the rest of the people on board and preliminary tests taken during the journey returned negative results, authorities said. 

"Spectrum of the Seas is taking appropriate measures under guidelines by the Department of Health," Royal Caribbean said in a statement. 

The ship was on early Wednesday ordered to return to the Kai Tak Cruise Terminal. The nine close contacts will be sent to a quarantine center, while the rest of the passengers and staff will have to undergo several compulsory tests in the coming days, the government said. 
"""""""""],
["""""
Hong Kong has seen a record low in the Joint University Programmes Admissions System this year, the lowest in nearly a decade.  

JUPAS - the main route to apply for local tertiary institutions - allows applicants to seek entry to full-time programs at the eight institutions funded by the University Grants Committee and the self-financed Hong Kong Metropolitan University.  

According to the JUPAS website, there were 38,955 applicants this year, a drop of 1,057 from last year. The figures have been declining each year since 2013 from the peak of 69,172.  

Reports suggested that the record figure could be a result of the city’s low birth rate and the increasing number of families moving abroad with their children, out of worries about the city’s political status quo.

It also noted that JUPAS updating its program list may also contribute to the drop in application numbers.
"""""]
]

io1 = gr.Interface.load('huggingface/sshleifer/distilbart-cnn-12-6')
io2 = gr.Interface.load("huggingface/facebook/bart-large-cnn")
io3 = gr.Interface.load("huggingface/csebuetnlp/mT5_multilingual_XLSum")  
io4 = gr.Interface.load("huggingface/google/pegasus-xsum")                   

iface = Parallel(io1, io2, io3, io4,
                 theme='huggingface', 
                 inputs = gr.inputs.Textbox(lines = 10, label="Text"), title=title, description=description, examples=examples)

iface.launch(share=False)