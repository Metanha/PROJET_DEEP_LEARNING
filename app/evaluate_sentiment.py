"""
evaluate_sentiment.py
-----------------------
Évaluation quantitative du sentiment, à deux niveaux :
1. Modèle de sentiment seul, sur du texte de référence propre (isole sa performance)
2. Pipeline complet (audio -> ASR -> sentiment), sur les fichiers de démo réels
   (mesure la performance réelle, incluant les erreurs de transcription)
"""

from sklearn.metrics import accuracy_score, f1_score

from .pipeline import SentimentPipeline
from .sentiment import SentimentModel

# --- Jeu de test texte propre (référence, sans audio) ---
TEXT_TEST_SET = [
    (
        "hi so i just wanted to call and say thank you i had an issue with my internet "
        "last week it kept dropping and a technician came the very next day he was really "
        "professional explained everything checked the router replaced a cable and since "
        "then everything has been perfect i wasn't expecting that kind of speed honestly "
        "the person who scheduled the visit was friendly too and she even sent a text to "
        "confirm the appointment i've been a customer for years and this might be the best "
        "experience i've had with your team so far thanks again",
        "positif",
    ),
    (
        "hello um yes i'm calling about my loan application and i just wanted to say the "
        "advisor who helped me was fantastic she explained every single fee clearly answered "
        "all my questions patiently and even called me back the same day with updates i was "
        "really nervous going into this because i've heard horror stories from friends about "
        "banks but this was smooth from start to finish i got approved faster than expected "
        "too so yeah just wanted to give some positive feedback because you don't often hear "
        "that",
        "positif",
    ),
    (
        "hi so my package arrived today actually a day early which surprised me and it was "
        "packed really carefully nothing was damaged the delivery driver even called ahead to "
        "make sure someone would be home which i really appreciated because last time with a "
        "different company it just got left outside in the rain the tracking updates were "
        "accurate the whole way through and customer service answered my question about the "
        "delivery window within minutes when i chatted online so overall a really pleasant "
        "experience thank you",
        "positif",
    ),
    (
        "yeah hi i wanted to call about my insurance claim from the accident last month i was "
        "honestly dreading this process because i assumed it would take forever but the agent "
        "walked me through everything step by step the payout was processed within a week and "
        "she checked in twice just to make sure i understood the next steps my car is already "
        "fixed and i'm back on the road so i really can't complain i was impressed given "
        "everything i'd heard about insurance claims being a nightmare",
        "positif",
    ),
    (
        "hi so i cancelled my subscription last month by mistake and called to see if it could "
        "be reactivated with my old settings and history the support agent not only restored "
        "everything perfectly but also applied a small credit for the inconvenience without me "
        "even asking she was patient explained the billing cycle clearly and confirmed "
        "everything by email right after the call this is exactly the kind of service that "
        "keeps me as a customer honestly",
        "positif",
    ),
    (
        "hello i'm calling to schedule my annual checkup and also to say thank you for how the "
        "front desk handled my appointment mix-up last time i had shown up on the wrong day "
        "because of a calendar error on my end and instead of turning me away they managed to "
        "fit me in the same afternoon the doctor was thorough answered all my questions and the "
        "whole visit felt unrushed which i really valued so i wanted to make sure that positive "
        "experience got noted somewhere",
        "positif",
    ),
    (
        "hi yes so i just landed and wanted to call about my flight that got rebooked due to "
        "weather the ground staff handled it really well they found me a seat on the next "
        "available flight within twenty minutes gave me a meal voucher without me asking and "
        "kept announcing updates so nobody was left guessing i travel a lot for work and this "
        "was honestly one of the smoothest disruption experiences i've had so i wanted to pass "
        "along some positive feedback to the team",
        "positif",
    ),
    (
        "hey um i wanted to call about my gym membership i had paused it for two months while "
        "traveling and the reactivation process was so simple one phone call and it was done "
        "no fees no hassle and the staff even remembered my usual class schedule and helped me "
        "rebook my spots for this week that kind of personal touch is rare these days so thank "
        "you i just wanted to say i'm happy i stuck with this gym",
        "positif",
    ),
    (
        "hi so i had a billing question about my energy plan and honestly i expected a "
        "complicated call but the representative explained the tiered pricing clearly showed "
        "me how to reduce my usage during peak hours and even signed me up for a lower rate "
        "plan that better matches my actual consumption she saved me a real amount on my "
        "monthly bill going forward and didn't try to upsell me on anything unnecessary really "
        "solid experience overall",
        "positif",
    ),
    (
        "hello i wanted to call about the reservation i made for last weekend everything went "
        "perfectly the table was ready exactly on time the staff accommodated a last minute "
        "request to add two more guests without any issue and the manager even came by to check "
        "if everything was alright which felt like a nice personal touch we celebrated a "
        "birthday there and it turned out even better than we expected so i just wanted to pass "
        "along how happy we were with the whole evening",
        "positif",
    ),

    (
        "yeah hi i've been on hold for forty minutes and this is the third time i'm calling "
        "about the same billing error my bill keeps showing an extra charge nobody can explain "
        "the first time i was told it would be removed within two cycles that never happened "
        "the second time i got transferred three times and the call just dropped now here i am "
        "again explaining the same thing i don't have hours to spend every month fixing your "
        "mistakes and honestly i'm considering canceling everything at this point",
        "négatif",
    ),
    (
        "hi so my flight got cancelled with less than two hours notice and nobody from your "
        "team proactively contacted me i had to find out by checking the app myself then i "
        "waited in line for over an hour just to be told the next available flight was two "
        "days later with no compensation offered upfront i had a business meeting i missed "
        "entirely because of this and when i asked about a refund for my hotel i was told to "
        "submit a form online with no guarantee of anything this whole situation has been "
        "handled terribly",
        "négatif",
    ),
    (
        "hello i'm calling about my insurance claim again it's been six weeks since the accident "
        "and i still haven't received any payout every time i call i get a different answer "
        "about what's missing from my file first it was a photo then it was a police report "
        "that i already submitted twice now they're asking for something else entirely meanwhile "
        "i'm paying out of pocket for repairs this is not what i was promised when i signed up "
        "for this coverage and i'm honestly losing patience with this whole process",
        "négatif",
    ),
    (
        "hi um so i ordered something two weeks ago and it still hasn't arrived the tracking "
        "hasn't updated in eight days when i contacted support the first time they said it was "
        "just delayed in transit the second time a different agent told me it was actually lost "
        "and offered a reship but now a third agent is telling me i need to wait fourteen more "
        "days before they can even open an investigation nobody seems to know what's actually "
        "happening and i've already paid for something i clearly don't have",
        "négatif",
    ),
    (
        "yeah hi my internet has been down for almost three days now i've called every single "
        "day and each time i'm told a technician will come within twenty four hours nobody has "
        "shown up i work from home so this is actually costing me money at this point i missed "
        "an important call with a client yesterday because of this and when i asked for some "
        "kind of compensation i was just told to wait for the next billing cycle this is "
        "honestly one of the worst support experiences i've had with any company",
        "négatif",
    ),
    (
        "hi so i cancelled my subscription two months ago and i'm still being charged every "
        "month i've called three times now each time i'm told it's been cancelled and that i'll "
        "get a refund for the extra charges and yet here i am again looking at another charge on "
        "my statement nobody can explain why this keeps happening and i'm starting to think this "
        "is not just a mistake anymore i want this resolved today because i'm done being told "
        "the same thing every single time",
        "négatif",
    ),
    (
        "hello i'm calling about my appointment that got cancelled without any notice i drove "
        "forty minutes to get there and was told at the front desk that the doctor wasn't even "
        "in today nobody called or texted to warn me beforehand this is actually the second time "
        "this has happened this year and both times i had to rearrange my entire work schedule "
        "for nothing i understand things come up but there has to be some kind of communication "
        "and right now there's clearly none",
        "négatif",
    ),
    (
        "hi so i went to use my gym membership this morning and my card was declined at the door "
        "even though i've been paying every month without fail i called and was told there was "
        "some kind of system error on their end that had apparently frozen several accounts but "
        "nobody had bothered to notify affected members beforehand i had to stand there "
        "embarrassed in front of other people while staff figured out what was going on this is "
        "not the first billing issue i've had here either",
        "négatif",
    ),
    (
        "yeah hi my energy bill this month is nearly double what it usually is and nobody can "
        "give me a clear explanation i was first told it was an estimated reading then told it "
        "was actually based on an actual meter reading which contradicts what i was told five "
        "minutes earlier on the same call i asked for someone to come check the meter in person "
        "and was told the earliest slot is in three weeks meanwhile i'm expected to just pay "
        "this inflated amount in the meantime which doesn't seem fair at all",
        "négatif",
    ),
    (
        "hi i made a reservation for eight people last week and confirmed it twice by phone when "
        "we arrived there was no table for us at all the staff seemed confused and said they had "
        "no record of the booking we ended up waiting almost an hour standing near the entrance "
        "with young kids in our group before they managed to squeeze us in at a much smaller "
        "table than requested nobody apologized properly and honestly this ruined what was "
        "supposed to be a special celebration for us",
        "négatif",
    ),

    (
        "hello good morning i'm calling because i recently moved and i wanted to update my "
        "billing address and phone number on file i also wanted to ask whether my current plan "
        "is still available at the new address or if i need to switch packages and when my next "
        "payment is due since i think i missed the email notification could you also tell me if "
        "a technician visit is required or if i can do a self installation and what documents i "
        "would need if i have to visit a store in person",
        "neutre",
    ),
    (
        "hi i'm calling to ask a few questions about refinancing my mortgage i wanted to know "
        "what the current interest rates look like compared to my existing loan what documents "
        "i would need to provide and roughly how long the whole process usually takes could you "
        "also explain if there are any early repayment fees on my current loan and whether i "
        "would need to redo a property valuation as part of this process",
        "neutre",
    ),
    (
        "hello i wanted to check the status of an order i placed last week could you tell me "
        "the expected delivery date and whether it ships in one package or several also do you "
        "know if i can change the delivery address at this point since i'll be traveling next "
        "week and finally can you confirm whether a signature is required upon delivery or if it "
        "can just be left at the door",
        "neutre",
    ),
    (
        "hi i'm calling about renewing my car insurance policy which i believe expires next "
        "month could you walk me through what coverage options are available this year and "
        "whether adding a second driver would significantly change the premium also i wanted to "
        "ask if there's a no claims discount applied automatically or if i need to request it "
        "specifically",
        "neutre",
    ),
    (
        "hello i wanted to ask about upgrading my current subscription plan to the next tier "
        "could you explain what additional features are included and whether the upgrade takes "
        "effect immediately or at the start of the next billing cycle also is there a way to "
        "try the higher tier temporarily before committing to it long term",
        "neutre",
    ),
    (
        "hi i'm calling to schedule a routine dental checkup for myself and my two children "
        "could you let me know what times are available next week and whether all three "
        "appointments could be booked back to back also do i need to bring any previous records "
        "if we're switching from a different clinic",
        "neutre",
    ),
    (
        "hello i wanted to ask about the baggage policy for an upcoming flight specifically how "
        "many checked bags are included in my fare class and what the weight limit is could you "
        "also confirm whether i can pre pay for an extra bag online instead of at the airport "
        "and what the price difference would be",
        "neutre",
    ),
    (
        "hi i'm calling about freezing my gym membership for two months while i'm traveling for "
        "work could you tell me if there's a fee for pausing and whether my membership start "
        "date shifts accordingly once i resume also can class bookings still be made during the "
        "freeze period or only after it ends",
        "neutre",
    ),
    (
        "hello i wanted some information about switching to a different energy tariff could you "
        "explain the difference between the fixed rate and variable rate options currently "
        "available and whether switching now would involve any exit fees from my current plan "
        "also how long does the switch usually take to go into effect",
        "neutre",
    ),
    (
        "hi i'm calling to ask about making a reservation for a group of ten people next month "
        "could you tell me if a deposit is required for groups that size and whether you can "
        "accommodate a specific dietary request for one of the guests also what's the latest "
        "time we could push the reservation to on a weekday evening",
        "neutre",
    ),
]

def evaluate_sentiment_model_only():
    model = SentimentModel()
    y_true, y_pred = [], []

    print("=== Modèle de sentiment seul (texte propre) ===")
    for text, expected in TEXT_TEST_SET:
        result = model.predict(text)
        y_true.append(expected)
        y_pred.append(result.label)
        marker = "OK" if result.label == expected else "FAUX"
        print(f"  [{marker}] attendu={expected:8s} prédit={result.label:8s} (conf={result.confidence:.2f}) | {text}")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"  -> accuracy={acc:.3f}  f1_macro={f1:.3f}\n")
    return acc, f1


def main():
    text_acc, text_f1 = evaluate_sentiment_model_only()


    #print("=== Comparaison ===")
    print(f"{'Modèle seul (texte propre)':35s} acc={text_acc:.3f}  f1={text_f1:.3f}")
    #print(f"{'Pipeline complet (audio réel)':35s} acc={pipeline_acc:.3f}  f1={pipeline_f1:.3f}")


if __name__ == "__main__":
    main()
    
