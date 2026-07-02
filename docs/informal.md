# Catching OSRS bots by watching them disappear

Old School RuneScape is a game whose central feature is, by deliberate design, that almost nothing in it is fast. 
Skills start at level 1 and end at level 99 along an exponential XP curve where reaching level 92 takes as much XP as the climb from 92 to 99. 
There are twenty five separate skills and hundreds of tracked activities; a single character takes thousands of hours to mature, and people play seriously for years.

There is also a mature and robust economy with a common currency, and many of the skills are trained during resource gathering in click-and-wait loops.
In-game items are exchanged for the game's currency, and third party black markets will convert that currency to real money, which means that someone running eight hundred mining accounts on cheap VMs is, in effect, running a small ore farm that produces dollars.
Jagex, the British studio operating the game, bans these accounts, often in waves, and has done so with varying degrees of success for over twenty years.

When an account is banned, it disappears from the game's public leaderboard (the Hiscores). PanoptOS is a methodology which leverages this fact to build a dataset for bot detection using entirely public information.
The disappearances act as free labels. With enough patience (and a sane crawl strategy), one can assemble a dataset of millions of OSRS accounts where every account is automatically labeled as banned or not by the game's own enforcement system.
Some disambiguation via RuneMetrics is needed to separate bans from things like name changes, but from there it's a fairly classical supervised learning problem.
However, the setting forces us to confront three main questions: where to look, how to decide whether an account is a bot, and how to tell when the model is good enough.

## Problem 1: where to look

The OSRS Hiscores aren't one list, they are hundreds, divided across three variants (the main Hiscores, plus separate boards for level-3 skillers and 1-Defence pures — accounts playing under self-imposed restrictions).
Each variant contains an overall leaderboard ordered by total cumulative XP, twenty five separate lists for the individual skills, and an ever-expanding set of (currently) approximately eighty five lists for in game activities like boss kill counts and minigame rankings.
Each of these lists is capped at two million users and overlaps heavily with the others, since an account that plays broadly will appear on dozens of the lists simultaneously.

The lists are not all equally useful for finding bots, though. Gathering skill tables like Mining and Woodcutting typically contain orders of magnitude more bots than endgame raid tables like Theatre of Blood,
because high volume gold farms favor cheap, repeatable, high throughput tasks over mechanically demanding fights.
A hypothetical crawler sampling from each Hiscore table at the same rate throws away a large portion of its request budget on tables that barely contain any of what it's looking for.
We want to be considerate and not absolutely hammer Jagex's API, so the budget needs to be spent efficiently. Out of every fixed batch of requests, where should they go?

This problem can be framed as a *multi-armed bandit* problem, named after a row of slot machines with unknown payout rates.
The classical version maps neatly onto ours: you have *K* machines (tables), each paying out at some unknown rate (the table's bot percentage), and a fixed number of pulls (requests per unit time).
You can't try each machine equally often, since that wastes pulls on bad ones, but you also can't commit early to a single machine, since it might be a bad one. There is an exploration-exploitation tradeoff.

PanoptOS solves this problem with a probability matching strategy using Bayesian utility estimates. For each table, the system maintains a probability distribution over that table's ban rate.
As bans land, the distribution updates, and its current average is the running best estimate of the table's true ban rate, discounted by an exponential recency bias.
The crawler picks tables in proportion to that estimate, mixed with a small uniform floor so every table keeps some guaranteed minimum share of the budget. Over time the budget naturally migrates towards
productive tables, and the result is that gathering skills like Woodcutting or heavily botted bosses like Callisto are weighted heavily and endgame tables like Sol Heredit feature lightly.

There's one wrinkle. A Hiscores table can be both a confirmed gold mine *and* completely saturated. After a long enough crawl every page the system fetches on a particular table
may be fully composed of accounts already in the database, especially if that table has a low churn rate, so the ban rate stays high but few new bots are discovered per fetch.
PanoptOS handles this by tracking a second rate per table, the discovery rate, which is the probability that a fetch from that table turns up an account the system hasn't seen before.
The final sampling weight is the product

> selection weight = (estimated ban rate) × (estimated discovery rate).

Thus, a table earns budget in proportion to the expected number of eventually banned, newly discovered accounts per fetch.
This decomposition is deliberate, and gives us easily interpretable knowledge on what the crawler policy is optimizing for. 
The discovery rate uses a similarly time-discounted scheme, so that the estimate reflects primarily the last few hundred fetches for a table rather than the entire history.
Saturated tables therefore have a built-in path to recovery as the underlying account population shifts.

There is a related but separate scheduling question. Once the crawler has found an account, when should it look at it again? Repeated querying
is needed not only to see whether the account has been banned, but to accumulate a series of snapshots of its history. 
Querying every account each day would saturate the API and be rude to Jagex, but querying each account once per month would miss most short-lived bots, since their entire bot phase may fit inside a single sampling interval.
PanoptOS uses an adaptive requery schedule that tightens for active accounts and stretches for dormant ones, weighted towards newer accounts (whose XP profile changes most rapidly). The base interval ranges from half a day to three and a half days,
with a multiplier that doubles as an account matures.

## Problem 2: how to decide

Once an account is being watched, we want to determine as quickly as possible if it is a bot.

For each account, the crawler sees a sequence of snapshots over time. Each snapshot is ~175 numbers describing the state of the account at that moment.
About 110 of those numbers are raw cumulative XP and activity counts, log transformed because XP totals span eight orders of magnitude.
Another 25 are velocities representing how fast the account gained XP in each skill since the last snapshot. 
Another 24 are ratios representing what fraction of the total XP currently sits in each skill, which helps make behavioral profiles comparable across early game and late game accounts.

On top of these, we engineer several features designed to add discriminative power to the model. A bot grinding Fishing for two weeks gains
XP almost entirely in one skill, so the *entropy* of its per snapshot gain vector sits near zero. Two consecutive snapshots, each describing an interval's worth of gains, look almost identical, because the bot is in a loop;
the cosine similarity between consecutive gain vectors is near 1.0. We also compute an activity streak, meaning the number of consecutive observation periods where the same single activity dominates. 
The velocity delta (change in total XP per hour from one interval to the next) sits near zero for unsophisticated bots, because they hold a steady pace.

A human grinding Fishing for those same two weeks looks meaningfully different. Humans go AFK, take breaks, switch to different skills for a few hours. They will get pulled into quests or log off mid task,
or get sucked into what YouTuber Marstead described best as OSRS' "Candy Loop" (ooo, a piece of candy!).
Their gain vectors are noisy and bursty, and even the most dedicated human players rarely match the robotic regularity of the most sophisticated bot scripts.
The most recent cumulative XP, which is all a single Hiscores snapshot directly shows, captures none of this. The interesting features compare snapshots to each other over time.

This is the core argument for using a sequence model rather than a simple classifier on the most recent snapshot.
A model trained on a single snapshot can split on "current Mining XP" or "share of XP in skill *k*", and it can go pretty far on those alone.
But a sequence model can ask those questions and more, like whether a high velocity preceded or followed a switch in dominant activity, whether two consecutive intervals look suspiciously alike, or even where in the observations a behavioral change point sits.
Those signals a single snapshot model cannot see, no matter how sophisticated it is or how good its human-constructed features are.

Here, we feed the ~175-number snapshots through a model architecture specifically designed for sequence data. When the model processes any
one snapshot, it can also look back at every earlier snapshot in the sequence and decide which of them are worth paying attention to, weighting them by relevance.
We keep the model purposefully small, because our goal is to run all of this comfortably on a Raspberry Pi.

![The PanoptOS architecture](https://i.imgur.com/B9OEBGx.png)

Since the snapshots are not equally spaced in time, the model should treat two snapshots taken three days apart differently from two taken three weeks apart, even if their contents are identical.
We solve this with a scheme called Fourier time encoding. Each snapshot's "days since the account was first seen" gets turned into a bundle of sines and cosines spanning different characteristic timescales.
The fastest completes a full cycle in about ten hours (our shortest snapshot window), whereas the slowest takes a year. Taken together, the bundle behaves like
a smooth fingerprint of when the snapshot occurred relative to discovery. Because the timescales span such a wide range, the model can read off both
fine grained differences (snapshot A is a few hours after snapshot B) and coarse grained ones (this account has been three-tick-teaking for weeks and then decided to do something less deranged).
Which timescales matter is left for the model to learn from the data.

In addition to the snapshots, the model has a second branch that operates over the account's name. 
OSRS account names are at most twelve characters long, and bot farms sometimes use formulaic ones with sequential numbers, random letter strings, or short sequences of random words.
This name branch contains a much smaller network than the snapshot branch, since twelve characters is a tiny information budget and a large network here would happily overfit to spurious patterns that happen to correlate with bans in the training distribution but which have
no predictive power on accounts the model hasn't seen.

The two branches combine through a gate, another small neural network that looks at the behavioral representation and decides how much to let the name contribute.
When the behavior is decisive, the name signal gets suppressed, but when the behavior is ambiguous the name weighs in more.
This helps guard against two failure modes: one is an account that just happens to have a bot sounding name (an *xSlayer420x* type) without actually being a bot.
The other is account takeovers, where an account previously run by a human is compromised through credential stuffing, phishing, or real world trade purchase, and switched to bot operation midlife.
In the takeover case, the name was chosen by the original owner and is misleading, and the gate is designed to suppress it.

## Problem 3: when is it good enough?

The standard way to evaluate a binary classifier is to give it all the input available at evaluation time, look at how well it ranks
positives over negatives, and report a metric like ROC-AUC, a number between zero and one where higher is better, 0.5 means useless, and 1.0 means perfect classification.
Under a "latest history" evaluation, where each account is scored on its most recent retained sequence (up to our maximum of 30 snapshots), PanoptOS reaches a ROC-AUC of about 0.995.
On its own that number is extremely impressive, but unfortunately it is almost useless as a measure of how good the system is.

To see why, we need to think about the objective we are training against. Our labels are Jagex bans, and a classifier that confidently identifies
bots at the same moment Jagex is banning them adds no incremental detection value. The window in which an external classifier can add value
is within the first few days after an account becomes observable, i.e. we want to confidently detect bots as early as possible.
The 0.995 number confirms that the model has learned the task when later history is available, but it does not answer the early detection question.

The third part of the methodology answers that question directly. Instead of the 0.995 latest history number, we report an AUC
as a function of how much calendar time has elapsed after account discovery. We test the model when it is allowed to see four days, five days, and so on out to thirty days of account history starting at the discovery date. 
The abridged curve looks like:


| Days observed | ROC-AUC |
|---|---|
| 4  | 0.981 |
| 7  | 0.982 |
| 14 | 0.985 |
| 30 | 0.986 |

ROC-AUC is already at 0.981 with four days of observation post-discovery, which means that the model is operationally useful right away and only improves as the account ages.
The thirty day number is 0.986 rather than 0.995 because it uses the first thirty days after discovery instead of the last thirty, which is a harder and more useful test. To capture all of this
into one number, we summarize the full curve into an integrated time AUC (itAUC), which is the area under the curve of AUC plotted against days observed, normalized to the horizon range.
It captures how quickly a model becomes useful, which is the property we care about. PanoptOS scores itAUC = 0.985 over days 4 through 30. Two models 
with identical late-history AUC can have very different itAUC, and it is the figure of merit here that maps to the real problem we are trying to solve.

![PanoptOS performance across observation windows after discovery. The useful signal appears within the first few days of an account's observable life.](https://i.imgur.com/sFI9wY8.png)

The benchmark for that number is the closest existing prior work and the only direct precedent: the *Bot Detector* RuneLite plugin, which 
is an open source, community led project that, like PanoptOS, doesn't require any access to Jagex internal data. 
Its production classifier is a tree based model that scores each candidate from a single Hiscores snapshot. While we don't have direct
access to Bot Detector's data, reproducing their methodology on the same source data and labels used to train PanoptOS makes for a clean head-to-head comparison:


| Metric    | PanoptOS | Bot Detector | Gap  |
|-----------|----------|--------------|------|
| itAUC-ROC | 0.985    | 0.950        | +3.5 |

The gap is concentrated at short horizons, which, as discussed, is where it matters most. Measured as an integrated ROC-AUC,
PanoptOS is about three and a half points ahead of the single snapshot reproduction. The gap widens to over six points when measured with the precision focused version of the metric (integrated PR-AUC), which weights exactly the high confidence regime an operator would act on. A snapshot taken late in an account's history carries 
as much information as a single instant can, but the trajectory closes the rest of the gap, and that gap is largest when there is no late snapshot to fall back on.

![PanoptOS compared with a single snapshot Bot Detector style baseline. The gap is largest when only a few days of history are available.](https://i.imgur.com/6HgHrfi.png)

## A few odds and ends
Although we train the model on a modern GPU, the deployment target for inference is a Raspberry Pi 5. To make this work, the trained model goes through int8 quantization,
in which the network's weights and biases, normally stored as 32-bit floating point numbers, are converted instead to 8-bit integers for computational efficiency.
The cost is negligible, and we see the performance measured above unchanged to three decimal places.

With that said, this approach has a few limitations worth mentioning. The first is the dataset, which is what the literature calls "positive-unlabeled".
Bots that Jagex never catches end up labeled as normal, which contaminates our negative class and biases the loss in ways that hurt potentially higher precision operation.
The crawler itself also only sees bots that survive long enough to reach a rank on at least one Hiscores table, so PanoptOS
targets the population that already evaded Jagex's faster detection methods. Although this shrinks our population, it has an upside in that these are exactly the bots human players are most likely to encounter in game.
More fundamentally, this entire methodology operates under the assumption that Jagex's enforcement is slow enough to leave a usable observation window. Faster enforcement on Jagex's end would, ironically for us, shrink that window and weaken our labels accordingly.

There are also entire categories of bot that PanoptOS can't see at all. Gambling bots and GE spammers repeating RWT advertisements never gain any meaningful XP or activity counts, and so never appear on any Hiscores table.
They leave no trail in this dataset, and detecting them would require a totally different signal involving scraping in-game chat which would likely violate Jagex TOS
and open us up to all sorts of privacy adjacent issues, so we do not make any attempt to do that here.

In the end, the signals the model relies on are unremarkable in isolation.
OSRS players do strange and remarkable grinds all the time, but they are still human beings and get distracted,
bank stand, change plans, or wander off into an enticing quest. But across enough snapshots, even subtle regularity resolves into a distinct rhythm.
For a more technical and detailed treatment of this problem, see the [accompanying report](report.pdf).