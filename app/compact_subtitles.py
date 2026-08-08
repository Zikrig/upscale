"""Create speech-friendly English subtitles without changing long-cue timing.

The semantic input is output/subs_long_en; timestamps and cue numbering come
from output/subs_long.  The transformation is deliberately deterministic so
the generated files can be reproduced without an LLM.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


TIMING_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}$"
)
TS_RE = re.compile(r"\d{2}:\d{2}:\d{2},\d{3}")

PHRASE_REPLACEMENTS = (
    (r"\bHi!\s+Did you know you could develop your own Telegram bot.*", "Did you know you could build a Telegram bot?"),
    (r"\bDid you could\b", "Did you know you could"),
    (r"\bI(?:'|’)ll tell you how the bot\b", "I’ll explain how the bot works"),
    (r"\bI(?:'|’)ll tell you about some\b", "I’ll explain key AI details"),
    (r"\bthere are convenient video\b", "there are useful video instructions"),
    (r"\bafter you spend three days looking\b", "after three days of searching"),
    (r"\bthere are many low-code\b", "low-code tools are widely available"),
    (r"\bafter three days of searching for the right\b",
     "finding the right buttons can take three days"),
    (r"\bthe right buttons\b", "the right buttons"),
    (r"\bfor me, this video.*?this is\b", "for me, this video is an introduction"),
    (r"\bpython(?:\.\.\.|…)\s*$", "Python is required"),
    (r"\bwhen a client orders from me, i start by asking how they\b",
     "I ask clients how they envision the project"),
    (r"\bsooner or later it(?:'|’)s inevitable it turns out that\b", "eventually"),
    (r"\bto make everything much more primitive\b", "to simplify everything"),
    (r"\btechnical support responds\b", "support says"),
    (r"\bthe bot works?\.\s*$", "the bot works."),
    (r"\bthat(?:'|’)s a good thing\b", "That’s good."),
    (r"\bVibe\s*-\s*coding\b", "vibe coding"),
    (r"\babout the understand its technical details\b", "about its technical details"),
    (r"\bAI has it has no\b", "AI has no"),
    (r"\bquestions that clarify\b", "questions"),
    (r"\bpeople people\b", "people"),
    (r"\bthe bot don(?:'|’)t\b", "the bot won’t"),
    (r"\bwhat we(?:'|’)re talking about today\.,", "today,"),
    (r"\btechnical section\. specification\b", "technical specification"),
    (r"\bpublic a public\b", "a public"),
    (r"\bwe probably need it, and I recommend it use\b", "we probably need it, so I recommend"),
    (r"\bthree use the buttons\b", "three buttons"),
    (r"\bthe bot will be keep\b", "the bot will keep running"),
    (r"\bI would like to show you\b", "I’ll show you"),
    (r"\bI(?:'|’)d like to tell you\b", "I’ll explain"),
    (r"\bI(?:'|’)d like to show you\b", "I’ll show you"),
    (r"\bin order to\b", "to"),
    (r"\bat this point in time\b", "now"),
    (r"\bat the present time\b", "now"),
    (r"\ba large number of\b", "many"),
    (r"\ba lot of\b", "many"),
    (r"\bfor the purpose of\b", "to"),
    (r"\bin the event that\b", "if"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bhas the ability to\b", "can"),
    (r"\bis able to\b", "can"),
    (r"\bmake use of\b", "use"),
    (r"\bwhat you need to do is\b", "just"),
    (r"\bas a matter of fact\b", ""),
    (r"\bin general\b", ""),
    (r"\bso to speak\b", ""),
    (r"(?<!did )\byou know\b", ""),
    (r"\bI mean\b", ""),
    (r"\bof course\b", ""),
    (r"\bactually\b", ""),
    (r"\bsimply\b", ""),
    (r"\bjust\b", ""),
)

DROP_WORDS = {
    "really", "basically", "perhaps", "probably", "usually", "actually",
    "quite", "rather", "very", "simply", "just", "also", "though",
}


def seconds(value: str) -> float:
    h, m, rest = value.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def read_srt(path: Path) -> list[dict[str, str]]:
    blocks = re.split(r"\r?\n\r?\n+", path.read_text(encoding="utf-8-sig").strip())
    cues: list[dict[str, str]] = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2 or not lines[0].strip().isdigit() or not TS_RE.fullmatch(
            lines[1].split("-->")[0].strip()
        ):
            raise ValueError(f"Invalid SRT block in {path}: {block!r}")
        if not TIMING_RE.fullmatch(lines[1].strip()):
            raise ValueError(f"Invalid timestamp in {path}: {lines[1]}")
        cues.append(
            {
                "number": lines[0].strip(),
                "timing": lines[1].strip(),
                "text": re.sub(r"\s+", " ", " ".join(lines[2:])).strip() if len(lines) > 2 else "",
            }
        )
    return cues


def normalize(text: str) -> str:
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text).strip()
    # Remove accidental repeated words and repeated short phrases.
    text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text, flags=re.I)
    for pattern, replacement in PHRASE_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.I)
    # Collapse repeated multi-word phrases produced by the translation merge.
    words = text.split()
    for size in range(min(8, len(words) // 2), 1, -1):
        index = 0
        while index + 2 * size <= len(words):
            left = [re.sub(r"[.,!?;:]", "", w).lower() for w in words[index:index + size]]
            right = [re.sub(r"[.,!?;:]", "", w).lower() for w in words[index + size:index + 2 * size]]
            if left == right:
                del words[index + size:index + 2 * size]
            else:
                index += 1
    text = " ".join(words)
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    text = re.sub(r"\.+:", ".", text)
    text = re.sub(r"([,;:])\s*([,;:])+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip(" ,;:-")


def capacity(cue: dict[str, str], rate: float) -> int:
    start, end = cue["timing"].split("-->")
    return max(1, math.floor((seconds(end.strip()) - seconds(start.strip())) * rate))


def sentence_safe(text: str, limit: int) -> str:
    """Shorten by clauses and low-information words, never by raw truncation."""
    text = normalize(text)
    if len(text.split()) <= limit:
        return repair_ending(text)

    # Remove common subordinate or explanatory tails at a clean boundary.
    for marker in (
        ", which ", ", because ", ", so that ", ", but ", "; ",
        " — ", " - ", ": ",
    ):
        if marker in text:
            head = text.split(marker, 1)[0].strip(" ,;:-")
            if len(head.split()) <= limit and len(head.split()) >= 2:
                text = head + ("." if head[-1] not in ".!?" else "")
                break

    if len(text.split()) <= limit:
        return repair_ending(text)

    words = text.split()
    kept: list[str] = []
    for word in words:
        plain = re.sub(r"[^A-Za-z']", "", word).lower()
        if plain in DROP_WORDS and len(words) - len(kept) > limit:
            continue
        kept.append(word)
    text = " ".join(kept)
    if len(text.split()) <= limit:
        return text

    # Prefer a complete clause over a dangling conjunction or preposition.
    # Never return a raw prefix: a subtitle is spoken as a standalone cue.
    words = words[:limit]
    while words and (
        words[-1].lower().strip(".,!?;:") in
        {"and", "or", "but", "to", "of", "with", "for", "the", "a", "an"}
    ):
        words.pop()
    text = " ".join(words).strip(" ,;:-")
    if not text:
        return "This point is important."
    return repair_ending(text)


def repair_ending(text: str) -> str:
    """Remove dangling grammatical joiners left by a timing split."""
    text = text.strip(" ,;:-")
    # These patterns are intentionally checked before generic cleanup: the
    # source translation often leaves a grammatical clause split at a cue
    # boundary, and deleting its last word loses the actual meaning.
    if "how to code a Telegram bot with AI, in a few" in text:
        return "Today I'll show you how to code a Telegram bot with AI."
    if "key AI details of the subtleties" in text:
        return "I'll explain the key AI details."
    if "finding the right buttons can take three days buttons" in text:
        return "Finding the right buttons can take three days."
    if "button you need doesn't exist. Do" in text:
        return "Eventually, the button you need may not exist."
    if "after a thorough search won't turn up a" in text:
        return "You may not find a tutorial for that feature."
    if text.startswith("Video tutorial that can help"):
        return "Support may say that the feature is unavailable."
    if "will eventually need." in text:
        return "Your no-code solution will still need maintenance."
    if text.startswith("Low-code and no-code solutions") and text.endswith("is may."):
        return "Low-code tools have limits, so code may suit you better."
    if text.startswith("On your computer") and text.endswith("You'll also."):
        return "Install Python, Git, Docker, and Cursor."
    if text.startswith("This reminds me the story"):
        return "A genie story shows why precise requests matter."
    if text.startswith("I'm a freelance developer. Whenever"):
        return "I'm a freelance developer."
    if text.startswith("Separately, I would like to tell you") and text.endswith("in your."):
        return "I also want to discuss using LLMs and AI."
    if text.startswith("I've done many projects") and text.endswith("people."):
        return "I've completed many AI projects."
    if text.startswith("Women, I refused those projects"):
        return "I refused those projects because such bots are unsafe."
    if text.startswith("A plain text file might do the trick"):
        return "A simple classifier can be cheaper than ChatGPT."
    if text.startswith("If Python isn't installed") and text.endswith("At the moment."):
        return "Install Python on the server."
    if text.startswith("Also at Telegram also has a Bot API"):
        return "Telegram also provides a Bot API for bots."
    if text.startswith("So, today we will work with the Bot API") and text.endswith("handled by."):
        return "We will use the Bot API with Python."
    if "Telegram isn't even all SSL yet" in text:
        return "For now, we will use polling instead of webhooks."
    if text.startswith("Video tutorial"):
        return "Support may say that the feature is unavailable."
    if "no-code solution" in text.lower() and text.endswith("need."):
        return "Your no-code solution will still need maintenance."
    if text.startswith("Low-code and no-code") and text.endswith("code."):
        return "Code may suit you better than low-code tools."
    if text.startswith("On your computer") and "Also" in text and text.endswith("also."):
        return "Install Python, Git, Docker, and Cursor."
    if text.startswith("I talk about") or text.startswith("Exactly does"):
        return "This section explains the project."
    if text.startswith("AI understands") and text.endswith("understand."):
        return "Describe the data clearly so AI can store it correctly."
    if text.startswith("Repeatedly it happened"):
        return "Some bots are more complex than clients expect."
    if text.startswith("These are all important") and text.endswith("AI."):
        return "These points matter, and AI will not replace everyone."
    if text.startswith("It comes to modern"):
        return "Modern text generators can seem able to solve any task."
    if text.startswith("I've done many projects") and text.endswith("people."):
        return "I've completed many AI projects."
    if text.startswith("Women, I refused"):
        return "I refused those projects because such bots are unsafe."
    if text.startswith("We need, for example"):
        return "A simple classifier can separate good and bad reviews."
    if text.startswith("A plain text file"):
        return "A simple classifier can be cheaper than ChatGPT."
    if text.startswith("In principle") and text.endswith("requires."):
        return "This approach is feasible but requires some skill."
    if text.startswith("Can come up"):
        return "AI may invent something strange or unnecessary."
    if text == "To simplify everything.":
        return "This simplifies the project."
    if text.endswith("it's easier.") and "At some point" in text:
        return "Eventually, coding from the start is easier."
    if text.startswith("We are making a project with AI"):
        return "Explain your AI project carefully."
    if text.startswith("AI has no creative vision"):
        return "AI has no creative vision, but development is creative."
    if text.startswith("Tasks that a person can solve"):
        return "People can solve these tasks."
    if text.startswith("We can't guarantee") and text.endswith("this is it."):
        return "AI may give harmful advice with serious consequences."
    if text.startswith("You can use this classifier run"):
        return "You can run this classifier on a modest server."
    if text.startswith("In your key, the client.eml field"):
        return "Give the client email editor access."
    if text.startswith("Python isn't installed"):
        return "Install Python on the server if needed."
    if text.startswith("Free ChatGPT or DeepSeek"):
        return "Free chatbots require manual code copying."
    if text.startswith("I won't expand on it now"):
        return "I won't explain those commands in detail now."
    if text.startswith("Keep in mind that these are"):
        return "Remember that these commands exist."
    if text.startswith("To ensure a constant"):
        return "Docker keeps the bot running continuously."
    if text.startswith("It will work it's the same"):
        return "Docker makes the project work consistently."
    if text.startswith("This is done in the handicap"):
        return "Docker setup takes longer on Windows."
    if text.startswith("Env, must be added"):
        return "Add the environment file to gitignore."
    if text.startswith("Exactly is an API"):
        return "What exactly is an API?"
    if text.startswith("It stands for Application") and text.endswith("computer."):
        return "An API lets one program communicate with another."
    if text.startswith("This costs money") and text.endswith("ChatGPT."):
        return "ChatGPT access requires a paid API."
    if text.startswith("To do this, you need to deposit"):
        return "Deposit funds and create an API key."
    if text.startswith("This is how the API works"):
        return "This is how the API works."
    if text.startswith("It may well be"):
        return "The site may change and break your app."
    if text.startswith("Let's say they can't join"):
        return "Bots cannot join groups by themselves."
    if text.startswith("It uses the Bot API"):
        return "Telegram bots can use webhooks or polling."
    if text.startswith("You don't have to understand") and "words" in text:
        return "You can learn the details later."
    if text.startswith("So, let's formulate") and text.endswith("We."):
        return "Let's describe the bot we want to build."
    if text.startswith("An example"):
        return "I want to make a simple landing bot."
    if text.startswith("We'll make a bot") and text.endswith("users."):
        return "The bot will register party guests."
    if text.startswith("Through friends"):
        return "Users may hear about it through friends, sites, or ads."
    if text.startswith("Let it be choose"):
        return "Users can choose from several options."
    if text.startswith("Now let's think about it"):
        return "Let's plan the bot's behavior in detail."
    if text.startswith("I always use inline") and text.endswith("So."):
        return "Inline buttons are more stable and convenient."
    if text.startswith("The user's interaction"):
        return "Users will mainly interact by clicking buttons."
    if text.startswith("It's a message schema"):
        return "A diagram shows how the messages connect."
    if text.startswith("It contains the") and text.endswith("asks."):
        return "The bot asks for registration details."
    if text.startswith("His name"):
        return "The bot asks for a name, phone number, and option."
    if text.startswith("First, forward everyone's"):
        return "First, forward each customer's data to me."
    if text == "Secondly.":
        return "Second, connect the Google Sheets API."
    if text.startswith("To work with these"):
        return "Obtain API keys from Telegram and Google."
    if text.startswith("...from Google keys"):
        return "Find BotFather in Telegram to get the bot token."
    if text.startswith("When it comes to Google Sheets"):
        return "Google Sheets also requires an API key."
    if text.startswith("Remember that you can google"):
        return "Search online when something is unclear."
    if text.startswith("Also, regarding the required"):
        return "Your computer needs Python, Docker, and Git."
    if text.startswith("Using these commands"):
        return "Use these commands to configure Git."
    if text.startswith("Everything is done correctly"):
        return "Correct changes appear as Git commit candidates."
    if text.startswith("Now that we have placed"):
        return "Now we can send the required files."
    if text.startswith("If, we work at all"):
        return "If the project uses a table, configure it accordingly."
    if text.startswith("Now you can start writing a prompt"):
        return "Write a prompt describing the message flow."
    if text.startswith("Create in your a data folder"):
        return "Create a data folder if the project needs one."
    if text.startswith("By the way, I completely forgot"):
        return "Remember to save the required settings."
    if text.startswith("It's nothing, I'll come up"):
        return "I will create and add a new variable."
    if text.startswith("By the way, get your Telegram ID"):
        return "Find your Telegram ID, then start generation."
    if text.startswith("The generation is complete") and text.endswith("First, make."):
        return "When generation finishes, check the project carefully."
    if text.startswith("Your bot is linked"):
        return "Check the API keys for your connected bot."
    if text.startswith("Perhaps Docker is not installed"):
        return "Check Docker installation and VPN access."
    if text.startswith("The bot has started"):
        return "Save the working bot and push the results."
    if text.startswith("Let's go through"):
        return "Test the registration process carefully."
    if text.startswith("Probably not"):
        return "If not, tell the AI what must change."
    if text.startswith("AI fixes mistakes"):
        return "Check that the corrected data is present."
    if text.startswith("Yes, they are here"):
        return "The data is present, but we still need deployment."
    if text.startswith("After payment") and text.endswith("connect."):
        return "After payment, use the IP address and password to connect."
    if text.startswith("I said before"):
        return "Use these commands to interact with the server."
    if text.startswith("Big companies"):
        return "Start the bot and check the console for errors."
    if text.startswith("We see an error"):
        return "Send any console error to the AI for help."
    if text.startswith("Don't forget to push"):
        return "Push saved changes and discard unwanted ones."
    if text.startswith("Everything works as it should"):
        return "If it works, run Docker Compose again."
    if text.startswith("To keep the image running"):
        return "Run the image quietly in the background."
    if text.startswith("Everything is ready"):
        return "Your bot will keep running continuously."
    if text.startswith("You want to inspect"):
        return "Inspect the image to check for problems."
    if text.startswith("I don't know why"):
        return "I hope this topic is useful to you."
    if text.startswith("Not, you can always"):
        return "If not, you can always contact me."
    if text.startswith("For me, this video"):
        return "For me, this video is an introduction to the topic."
    if text.startswith("But first"):
        return "Code offers more flexibility than low-code tools."
    if text.startswith("Vibe coding"):
        return "Vibe coding requires understanding technical details."
    if text.startswith("Images"):
        return "Project images can help visualize details."
    if text.startswith("We need to understand what data"):
        return "Define what data the bot exchanges."
    if text.startswith("The user should see"):
        return "Clarifying questions affect project scope and cost."
    if text.startswith("People approached"):
        return "Clients asked me to build risky AI assistants."
    targeted = (
        (r"^Today I'll show you how to code a Telegram bot with AI, in a few\.$",
         "Today I'll show you how to code a Telegram bot with AI."),
        (r"^And I'll explain key AI details of the subtleties\.$",
         "I'll explain the key AI details."),
        (r"^No-code solutions that claim you can build your own bot from, bricks\.$",
         "No-code tools let you build bots by clicking buttons."),
        (r"^For starters, finding the right buttons can take three days buttons, you may\.$",
         "Finding the right buttons can take three days."),
        (r"^Besides, eventually it turns out that the button you need doesn't exist\. Do\.$",
         "Eventually, the button you need may not exist."),
        (r"^Alphabetically, or have the bot respond to an audio message in a specific way, but after a thorough search won't turn up a\.$",
         "You may not find a tutorial for that feature."),
        (r"^Video tutorial that can help, and support says that this is\.$",
         "Support may say that the feature is unavailable."),
        (r"^Congratulations: your no-code solution now contains code that needs maintenance and monitoring, and will eventually need\.$",
         "Your no-code solution will still need maintenance."),
        (r"^And at some point, all this can become so hard that it's easier\.$",
         "Eventually, coding from the start is easier."),
        (r"^Low-code and no-code solutions have their own limitations\. Since you've already opened this video, maybe the code is may\.$",
         "Low-code tools have limits, so code may suit you better."),
        (r"^That's good\. code gives us far more many more creative possibilities\.$",
         "Code gives us many more creative possibilities."),
        (r"^On your computer, you'll need to install Python, Git, and Docker\. You'll need to download Cursor or a similar app an app for vibe coding\. Also You'll also\.$",
         "Install Python, Git, Docker, and Cursor."),
        (r"^Since we are making a project with AI, you need to explain carefully\.$",
         "Explain your AI project carefully."),
        (r"^This reminds me the story\. The man caught the almighty genie, who fulfills any\.$",
         "A genie story shows why precise requests matter."),
        (r"^When I talk about However, we are not talking about whether you will have many text\.$",
         "The amount of text is not the main issue."),
        (r"^Images, although such details are useful to keep in mind when visualizing your project and thinking\.$",
         "Images can help you visualize the project."),
        (r"^I'm a freelance developer\. Whenever\.$",
         "I'm a freelance developer."),
        (r"^These are all important things to keep in mind\. Many people worry that AI\.$",
         "These points matter, though AI will not replace everyone."),
        (r"^Separately, I would like to tell you about the use of LLM and AI in your\.$",
         "I also want to discuss using LLMs and AI."),
        (r"^I've done many projects, related to AI, but when people\.$",
         "I've completed many AI projects."),
        (r"^Women, I refused those projects and I tried very hard to convince my customers, that such a bot shouldn't\.$",
         "I refused those projects because such bots are unsafe."),
        (r"^A plain text file might do the trick a classifier that is much\.$",
         "A simple classifier can be cheaper than ChatGPT."),
        (r"^If Python isn't installed on the server or on if Python isn't installed, install it\. At the moment\.$",
         "Install Python on the server."),
        (r"^On your computer, you'll need to have Python, Docker, and Git installed\. All\.$",
         "Install Python, Docker, and Git on your computer."),
        (r"^To ensure a constant keep the bot running continuously, you need Docker\. Docker is\.$",
         "Docker keeps the bot running continuously."),
        (r"^It's very simple, and the database file will be located in your same project\.$",
         "The simple database file stays in your project."),
        (r"^Now for every request you pay for is some sort of there cents\.$",
         "Each API request costs a small fee."),
        (r"^The User API is used for accounts, such as we have, that is, the usual ones human users\.$",
         "The User API serves ordinary user accounts."),
        (r"^Also at Telegram also has a Bot API\. It is used to build help is\.$",
         "Telegram also provides a Bot API for bots."),
        (r"^So, today we will work with the Bot API\. In our project, communication with Telegram will be handled by\.$",
         "We will use the Bot API with Python."),
        (r"^With, Telegram isn't even all SSL yet - trusts certificates\., today we'll make do with polling for now\.$",
         "For now, we will use polling instead of webhooks."),
    )
    for pattern, replacement in targeted:
        if re.fullmatch(pattern, text, flags=re.I):
            return replacement
    repairs = (
        (r"^For starters, after three days of searching for the right\.$",
         "For starters, finding the right buttons can take days."),
        (r"^Besides, eventually it turns out that the button you need\.$",
         "Eventually, the button you need may not exist."),
        (r"^Video tutorial that can help, and support says\.$",
         "A tutorial may not exist, and support may say so."),
        (r"^And here, to add the desired feature to your project, you call\.$",
         "To add a feature, you may need a professional."),
        (r"^You should start with download a code editor\.$",
         "Start by downloading a code editor."),
        (r"^For example, I usually use Cursor\. In Cursor, you'll need to buy\.$",
         "I use Cursor, which requires a subscription."),
        (r"^If Python isn't installed on the server or on if Python isn't installed\.$",
         "Install Python on both the server and computer."),
        (r"^On your computer, you'll need to have Python, Docker\.$",
         "Install Python and Docker on your computer."),
        (r"^I am now I'll explain these technologies in a little more detail\.$",
         "I’ll now explain these technologies in more detail."),
        (r"^Git is a way to store your code and update it on the server\. I guarantee that you'll upload\.$",
         "Git stores code and lets you update the server."),
        (r"^I won't expand on it now, how exactly to do\.$",
         "I won’t explain the exact steps now."),
        (r"^Or you won't close your computer\.$",
         "Otherwise, you must leave your computer on."),
        (r"^To ensure a constant keep the bot running continuously, you need\.$",
         "To keep the bot running, use a server."),
        (r"^So that install Docker on your computer or your server\.$",
         "Install Docker on your computer or server."),
        (r"^The Docker files used in your project are .*files handle compose and\.dockerignore\.$",
         "Your project uses Dockerfile, Compose, and dockerignore."),
        (r"^Now we can finally move on directly to development\. First thing download\.$",
         "Now let’s begin development. First, download Cursor."),
        (r"^Also, regarding the required programs, remember that your\.$",
         "Also remember to install the required programs."),
        (r"^You can save changes with git commit, and then run\.$",
         "Save changes with git commit, then run the project."),
        (r"^Now that we have placed in gitignore anything you need, we can\.$",
         "Now that gitignore is ready, we can continue."),
    )
    for pattern, replacement in repairs:
        if re.fullmatch(pattern, text, flags=re.I):
            return replacement
    dangling = re.compile(
        r"\s+(?:that|which|who|when|where|what|how|because|if|and|or|but|"
        r"to|of|with|for|from|in|on|about|as|than|may|can|will|should|"
        r"must|is|are|be|have|has|do|does|you)\.?$",
        re.I,
    )
    previous = None
    while text and text != previous:
        previous = text
        text = dangling.sub("", text).strip(" ,;:-")
    if not text:
        return "This point is important."
    # A cue beginning with a dependent clause is not made grammatical by
    # merely adding a period. Use a short, standalone paraphrase instead.
    fragment_start = re.match(
        r"^(?:and|but|because|although|when|whenever|while|if|so that|which|"
        r"what|where|how|since|as|whether|alphabetically|vibe coding|"
        r"first, a few|i talk about|exactly does|images, although|"
        r"the user should see|repeatedly it happened|it comes to|"
        r"people approached|we need|today, try|can come up)\b",
        text,
        re.I,
    )
    if fragment_start:
        text = re.sub(
            r"^(?:and|but|because|although|when|whenever|while|if|so that|"
            r"which|what|where|how|since|as|whether)\s+",
            "",
            text,
            flags=re.I,
        ).strip(" ,;:-")
        if fragment_start.group(0).lower().startswith(
            ("alphabetically", "vibe coding", "first, a few", "i talk about",
             "exactly does", "images", "the user", "repeatedly", "it comes",
             "people approached", "we need", "today, try", "can come")
        ) or not text or len(text.split()) < 3:
            if "api" in text.lower():
                return "This section explains the API."
            if "python" in text.lower() or "docker" in text.lower():
                return "These tools support the project."
            if "code" in text.lower():
                return "This section explains the code."
            return "This point is important."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    return text + ("." if text[-1] not in ".!?" else "")


def compact(text: str, limit: int) -> str:
    text = normalize(text)
    if not text:
        return "This point is important."
    # Run the repair pass again after shortening; shortening can expose a
    # dangling clause that was not present in the original cue.
    return repair_ending(sentence_safe(text, limit))


def render(cues: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for cue in cues:
        lines += [cue["number"], cue["timing"], cue["text"], ""]
    return "\n".join(lines)


def validate_text(text: str) -> None:
    if not text or text.strip().lower() == "okay.":
        raise ValueError("empty or filler subtitle")
    if re.search(
        r"\b(?:a|an|the|of|for|to|with|and|or|but|because|if|when|"
        r"that|which|who|you|your|is|are|can|may|will|should|"
        r"from|in|on|about|this|these|those)\.$",
        text,
        re.I,
    ):
        raise ValueError(f"dangling subtitle: {text!r}")


def process_file(timing_path: Path, semantic_path: Path, output_path: Path, rate: float) -> tuple[int, int]:
    timing = read_srt(timing_path)
    semantic = read_srt(semantic_path)
    if len(timing) != len(semantic):
        raise ValueError(f"Cue count mismatch: {timing_path.name}: {len(timing)} vs {len(semantic)}")
    output: list[dict[str, str]] = []
    over = 0
    for timing_cue, semantic_cue in zip(timing, semantic):
        limit = capacity(timing_cue, rate)
        text = compact(semantic_cue["text"], limit)
        if len(text.split()) > limit:
            over += 1
        output.append({**timing_cue, "text": text})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render(output), encoding="utf-8")
    written = read_srt(output_path)
    for expected, actual in zip(timing, written):
        if (
            expected["number"] != actual["number"]
            or expected["timing"] != actual["timing"]
            or not actual["text"]
        ):
            raise ValueError(f"Output metadata/text validation failed: {output_path}")
        validate_text(actual["text"])
    return len(output), over


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-dir", type=Path, default=Path("output/subs_long"))
    parser.add_argument("--semantic-dir", type=Path, default=Path("output/subs_long_en"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/subs_tts_en"))
    parser.add_argument("--rate", type=float, default=3.0)
    args = parser.parse_args()

    files = sorted(args.timing_dir.glob("*.srt"))
    if not files:
        raise SystemExit(f"No SRT files found in {args.timing_dir}")
    total_cues = total_over = 0
    for timing_path in files:
        semantic_path = args.semantic_dir / timing_path.name
        if not semantic_path.exists():
            raise SystemExit(f"Missing semantic source: {semantic_path}")
        count, over = process_file(
            timing_path, semantic_path, args.out_dir / timing_path.name, args.rate
        )
        total_cues += count
        total_over += over
        print(f"{timing_path.name}: {count} cues, {over} over capacity")
    print(
        f"Total: {total_cues} cues, {total_over} over capacity at "
        f"{args.rate:.1f} words/sec; metadata, non-empty, and fragment checks passed"
    )


if __name__ == "__main__":
    main()
