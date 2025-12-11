#!/usr/bin/env python3
"""
Generate 252 additional high-quality test cases using LLM-assisted reasoning
"""

import json
from pathlib import Path

# Uncertainty and confidence markers
UNCERTAINTY_MARKERS = [
    "i think", "i believe", "maybe", "probably", "possibly",
    "i'm not sure", "it seems like", "i guess", "presumably",
    "in my opinion", "i suppose", "might be", "could be", "perhaps"
]

CONFIDENCE_MARKERS = [
    "i know", "definitely", "certainly", "obviously", "clearly",
    "without a doubt", "absolutely", "i'm certain", "it's a fact",
    "undeniably", "for sure", "guaranteed", "no question"
]

def compute_scores(text):
    """Compute uncertainty and confidence scores"""
    text_lower = text.lower()
    uncertainty_count = sum(1 for marker in UNCERTAINTY_MARKERS if marker in text_lower)
    confidence_count = sum(1 for marker in CONFIDENCE_MARKERS if marker in text_lower)

    uncertainty_score = min(uncertainty_count / 3.0, 1.0)
    confidence_score = min(confidence_count / 3.0, 1.0)

    return uncertainty_score, confidence_score

# Generate 252 additional test cases (31-32 per emotion)
# Starting from existing test IDs (calm_007, confident_007, etc.)

additional_tests = []

# === CALM (31 more: 007-037) ===
calm_samples = [
    ("The death penalty debate requires considering both justice and the risk of wrongful execution. Historical data and ethical frameworks both offer important perspectives.", "criminal justice"),
    ("Looking at drug legalization objectively, we need to weigh public health outcomes, criminal justice impacts, and personal freedom. Portugal's data provides one interesting case study.", "drug policy"),
    ("College education costs present a complex policy challenge. We should examine both the value of degrees and the burden of student debt when discussing solutions.", "education policy"),
    ("Privacy versus security involves genuine tradeoffs. Perhaps we can find approaches that protect both interests through thoughtful policy design and technical safeguards.", "privacy"),
    ("The debate about social media age limits has merit on both sides. We need to consider child development research alongside practical enforcement challenges.", "social media"),
    ("When discussing tax reform, it's useful to look at economic models, historical precedents, and distributional impacts. Different proposals have different tradeoffs.", "tax policy"),
    ("Artificial intelligence regulation requires balancing innovation with safety concerns. We might learn from how other transformative technologies were governed.", "AI policy"),
    ("The question of reparations involves historical injustice, practical feasibility, and various policy approaches. It warrants careful examination of all perspectives.", "social justice"),
    ("Corporate governance reform presents interesting questions about stakeholder interests, accountability, and economic efficiency. Various models exist internationally.", "business policy"),
    ("When evaluating cryptocurrency regulation, we should consider financial stability, consumer protection, and innovation. The technology is still evolving rapidly.", "finance"),
    ("Antitrust policy in tech requires understanding network effects, competition dynamics, and consumer welfare. Historical monopoly cases offer some guidance.", "tech policy"),
    ("The debate over standardized testing in education has reasonable arguments on multiple sides regarding equity, accountability, and educational quality.", "education"),
    ("Prison reform involves balancing public safety, rehabilitation, and fiscal responsibility. Different countries have tried various approaches with mixed results.", "criminal justice"),
    ("When discussing infrastructure investment, we need to weigh economic benefits, environmental impacts, and financing mechanisms. The analysis requires nuance.", "infrastructure"),
    ("Rural broadband policy requires considering economic development, technical challenges, and public investment. Urban-rural equity is an important factor.", "technology access"),
    ("The question of space exploration funding involves scientific value, economic spinoffs, and opportunity costs. NASA's budget is worth thoughtful analysis.", "space policy"),
    ("Agricultural subsidies present complex questions about food security, environmental impacts, and international trade. The policy has evolved over decades.", "agriculture"),
    ("When evaluating election reform proposals, we should examine voting access, election security, and administrative feasibility. Different states offer natural experiments.", "voting policy"),
    ("Mental health parity in insurance requires balancing coverage, costs, and treatment access. The implementation has faced various challenges.", "healthcare"),
    ("The gig economy regulation debate involves worker protections, business models, and economic flexibility. Different jurisdictions have taken different approaches.", "labor policy"),
    ("Climate adaptation versus mitigation both deserve attention in environmental policy. Resources are limited, so allocation decisions matter.", "climate policy"),
    ("When discussing housing policy, we need to consider affordability, supply constraints, and local control. Zoning reform is one possible lever among many.", "housing"),
    ("The debate about nuclear weapons nonproliferation involves security dilemmas, verification challenges, and geopolitical dynamics. Arms control treaties have had mixed success.", "foreign policy"),
    ("Pandemic preparedness requires weighing probability, severity, and cost of interventions. Different countries had different strategies with COVID-19.", "public health"),
    ("The question of net neutrality involves competition, innovation, and market structure in telecommunications. Technical and economic factors both matter.", "internet policy"),
    ("When evaluating trade policy, we should consider efficiency gains, distributional effects, and strategic industries. Free trade agreements have complex impacts.", "trade"),
    ("Monetary policy independence requires careful institutional design. Different central banks have different mandates and accountability structures.", "economic policy"),
    ("The debate over affirmative action involves diversity goals, fairness concerns, and legal precedents. Court decisions have shaped the policy landscape.", "education equity"),
    ("When discussing border security, we need to consider enforcement effectiveness, humanitarian concerns, and resource allocation. Multiple policy tools are available.", "immigration"),
    ("Media literacy education presents interesting questions about curriculum design, digital citizenship, and information quality. Several countries have implemented programs.", "education"),
    ("The question of facial recognition regulation involves privacy, security applications, and accuracy concerns. Bias in algorithms is one important consideration.", "technology")
]

for i, (text, topic) in enumerate(calm_samples, start=7):
    uncertainty, confidence = compute_scores(text)
    additional_tests.append({
        "id": f"test_calm_{i:03d}",
        "text": text,
        "topic": topic,
        "true_emotion": "calm",
        "true_emotion_idx": 0,
        "true_uncertainty": uncertainty,
        "true_confidence": confidence
    })

# === CONFIDENT (31 more: 007-037) ===
confident_samples = [
    ("I'm absolutely certain that renewable energy is the future. The cost curves clearly show solar and wind are now cheaper than fossil fuels. The transition is inevitable.", "energy"),
    ("There's no question that income inequality has grown dramatically. The data is unambiguous - the wealth gap is at historic levels and definitely needs addressing.", "economics"),
    ("I know for a fact that vaccinations are one of humanity's greatest achievements. The evidence is overwhelming - they've saved millions of lives without a doubt.", "public health"),
    ("It's crystal clear that infrastructure investment pays for itself. Every economic study shows the multiplier effect. This is obviously a smart use of public funds.", "economics"),
    ("Educational technology is clearly revolutionizing learning. The personalization it enables is definitely improving outcomes. I'm certain this trend will accelerate.", "education"),
    ("Democracy is without question the best system of government. History proves that democratic nations are more stable, prosperous, and peaceful. This is undeniable.", "political systems"),
    ("I'm 100% sure that biodiversity loss threatens human survival. The scientific consensus is overwhelming. We definitely must protect ecosystems now.", "environment"),
    ("It's obvious that remote work increases productivity for knowledge workers. The data from countless studies clearly supports this. Companies should definitely embrace it.", "workplace"),
    ("Criminal justice reform is absolutely necessary. The evidence shows rehabilitation works better than punishment. This is clearly the right direction.", "justice"),
    ("I'm absolutely convinced that early childhood education has massive returns. Every longitudinal study proves it. Obviously we should invest more here.", "education"),
    ("There's no doubt that privacy is a fundamental right. The framers clearly understood this. Technology doesn't change that basic principle.", "civil rights"),
    ("I know with certainty that propaganda threatens democracy. History proves this repeatedly. We must definitely teach media literacy.", "education"),
    ("It's undeniable that smoking causes cancer. The medical evidence is absolutely conclusive. Tobacco regulation is clearly justified.", "public health"),
    ("I'm certain that infrastructure decay costs far more than maintenance. The engineering data proves this. Obviously prevention is smarter than repair.", "infrastructure"),
    ("Financial literacy is clearly essential for everyone. The data shows people make better decisions with education. This should definitely be taught in schools.", "education"),
    ("I know for certain that gerrymandering undermines democracy. The maps clearly show partisan manipulation. Obviously we need independent redistricting.", "voting"),
    ("It's crystal clear that affordable housing is an economic issue. Supply and demand work here like anywhere. We definitely need to build more.", "housing"),
    ("I'm absolutely sure that antibiotic resistance is a serious threat. The science is unambiguous. We clearly need better stewardship now.", "medicine"),
    ("There's no question that investment in research pays enormous dividends. Every analysis shows the returns. Obviously we should fund more.", "science policy"),
    ("I'm certain that financial transparency reduces corruption. The evidence from open government initiatives clearly proves this.", "governance"),
    ("It's definitely true that exercise prevents disease. The medical research is overwhelming. Obviously healthcare should emphasize prevention.", "public health"),
    ("I know without doubt that early intervention helps struggling students. The education research clearly shows this. We definitely need more support services.", "education"),
    ("It's absolutely clear that renewable energy creates more jobs than fossil fuels. The employment data proves it. This transition clearly benefits workers.", "energy"),
    ("I'm certain that voting access is fundamental to democracy. History clearly shows this. Obviously we should make it easier, not harder.", "voting rights"),
    ("There's no doubt that pollution has enormous health costs. The epidemiological evidence is overwhelming. Clean air regulation is clearly worth it.", "environment"),
    ("I know for a fact that food deserts harm public health. The research clearly links access to nutrition. Obviously we need better distribution.", "public health"),
    ("It's crystal clear that cybersecurity requires investment. The threat landscape obviously demands it. Organizations definitely need better defenses.", "security"),
    ("I'm absolutely convinced that parental leave benefits everyone. The social science clearly shows positive outcomes. This is obviously good policy.", "family policy"),
    ("There's no question that corruption undermines development. Economic studies prove this repeatedly. Transparency is clearly essential.", "governance"),
    ("I know with certainty that misinformation spreads faster than truth. The research is unambiguous. Obviously we need better digital literacy.", "media"),
    ("It's definitely true that infrastructure enables economic growth. Every development study shows this. Obviously we should invest strategically.", "economics")
]

for i, (text, topic) in enumerate(confident_samples, start=7):
    uncertainty, confidence = compute_scores(text)
    additional_tests.append({
        "id": f"test_confident_{i:03d}",
        "text": text,
        "topic": topic,
        "true_emotion": "confident",
        "true_emotion_idx": 1,
        "true_uncertainty": uncertainty,
        "true_confidence": confidence
    })

# === DEFENSIVE (32 more: 007-038) ===
defensive_samples = [
    ("That's not what I said about immigration policy at all. You're completely misrepresenting my argument. I was talking about legal pathways, not closing borders.", "immigration"),
    ("Actually, if you'd been listening, I clearly stated I support environmental protection. You're twisting my concerns about implementation into opposition.", "environment"),
    ("In my defense, I was pointing out the complexity of healthcare reform, not arguing against coverage. You're oversimplifying what I said.", "healthcare"),
    ("You're misunderstanding my point about education funding. I support teachers - I'm just questioning administrative overhead. That's different from what you're claiming.", "education"),
    ("Look, I already explained that I'm not against regulation. I'm concerned about unintended consequences. Why are you still suggesting I don't care about safety?", "regulation"),
    ("That's not fair at all. I clearly said that both perspectives have merit. You're making it sound like I dismissed one side completely.", "general debate"),
    ("Actually, you're mischaracterizing my position on gun policy. I said we need to balance rights with safety, not that we should do nothing.", "gun policy"),
    ("In my defense, I was trying to present a nuanced view of trade policy. You're making it seem like I'm dogmatically pro or anti when I'm neither.", "trade"),
    ("You don't understand what I'm saying about criminal justice. I support reform AND public safety. Those aren't mutually exclusive despite how you're framing it.", "justice"),
    ("That's not what I meant about technology regulation at all. You're putting words in my mouth. I said we need thoughtful rules, not no rules.", "tech policy"),
    ("Look, I clearly stated that I see both the benefits and risks. You're ignoring the nuance in my position and attacking a strawman.", "general debate"),
    ("Actually, I was making a point about implementation challenges, not opposing the goal. You're conflating criticism of means with rejection of ends.", "policy"),
    ("In my defense, that comment was about timing, not merit. You're taking it out of context to make me sound unreasonable.", "general debate"),
    ("You're misrepresenting what I said about welfare policy. I support helping people - I'm questioning whether this specific approach works best.", "welfare"),
    ("That's not fair. I acknowledged the data you presented. I was just adding context, not denying facts. Why are you treating that as disagreement?", "general debate"),
    ("Actually, I've been consistent throughout this discussion. You're claiming I changed positions when I clearly haven't. Check what I said earlier.", "general debate"),
    ("Look, I'm not defending that at all. You're attributing someone else's argument to me. I never said anything close to that.", "general debate"),
    ("In my defense, I was responding to a specific claim, not making a broad statement. You're expanding what I said beyond its actual scope.", "general debate"),
    ("You're misunderstanding the point I was making about foreign policy. I support diplomacy - I'm questioning this particular strategy.", "foreign policy"),
    ("That's not what I meant about economic policy at all. You're interpreting my comments in the most uncharitable way possible.", "economics"),
    ("Actually, I clearly stated my caveats when I made that point. You're ignoring them to make my position sound more extreme than it is.", "general debate"),
    ("Look, I already addressed that counterargument. You're acting like I ignored it when I specifically responded to it two minutes ago.", "general debate"),
    ("In my defense, I was trying to acknowledge complexity, not avoid the question. You're treating nuance as evasion.", "general debate"),
    ("You're putting me in a false binary. I don't have to choose between those two extremes - my actual position is more moderate.", "general debate"),
    ("That's not fair at all. I've been engaging with your points. Just because I don't agree doesn't mean I'm not listening.", "general debate"),
    ("Actually, if you look at what I actually wrote, I never said that. You're responding to something I didn't claim.", "general debate"),
    ("Look, I'm trying to have a good-faith discussion. You're attributing malicious motives when I'm genuinely trying to understand the issue.", "general debate"),
    ("In my defense, I admitted I don't have all the answers. You're treating that intellectual honesty as weakness.", "general debate"),
    ("You're conflating my skepticism about one proposal with opposition to the entire goal. Those are completely different things.", "general debate"),
    ("That's not what I said about privacy policy at all. You're extrapolating one comment into a comprehensive position I never took.", "privacy"),
    ("Actually, I was being precise in my language for a reason. You're treating my qualifications as hedging when they're important distinctions.", "general debate"),
    ("Look, I already clarified what I meant. You keep responding to my initial phrasing instead of my explanation. That's not productive.", "general debate")
]

for i, (text, topic) in enumerate(defensive_samples, start=7):
    uncertainty, confidence = compute_scores(text)
    additional_tests.append({
        "id": f"test_defensive_{i:03d}",
        "text": text,
        "topic": topic,
        "true_emotion": "defensive",
        "true_emotion_idx": 2,
        "true_uncertainty": uncertainty,
        "true_confidence": confidence
    })

# === DISMISSIVE (32 more: 007-038) ===
dismissive_samples = [
    ("Oh please. Anyone who thinks cryptocurrency will replace the dollar doesn't understand monetary policy. This is basic economics.", "currency"),
    ("That's absurd. The idea that we can eliminate all nuclear weapons is pure fantasy. Welcome to the real world.", "foreign policy"),
    ("Come on. Thinking social media can be regulated effectively shows a complete misunderstanding of technology. This is naive.", "tech policy"),
    ("Whatever. Anyone who believes in trickle-down economics at this point is ignoring decades of evidence. Use your brain.", "economics"),
    ("That's ridiculous. The notion that all jobs can be done remotely is detached from reality. Some work requires physical presence, obviously.", "workplace"),
    ("Please. You can't seriously think defunding police would work. That's been tried and it failed spectacularly. Think it through.", "policing"),
    ("Oh sure. Because completely open borders is totally practical. That's such a serious policy proposal. Not.", "immigration"),
    ("That's laughable. Anyone who thinks the free market solves every problem needs to read some history. Pure ideology.", "economics"),
    ("Come on now. The idea that everyone can just learn to code is Silicon Valley delusion at its finest. So out of touch.", "tech industry"),
    ("Whatever. Thinking thoughts and prayers solve gun violence is beyond naive. It's willful blindness to reality.", "gun policy"),
    ("That's nonsense. Anyone who denies climate change at this point is either ignorant or lying. The science is settled.", "climate"),
    ("Oh please. The notion that tax cuts pay for themselves has been debunked over and over. How do people still believe this?", "fiscal policy"),
    ("That's absurd. Thinking blockchain solves every problem is peak tech hype. Most use cases don't need it.", "technology"),
    ("Come on. Anyone who thinks the gender pay gap is a myth hasn't looked at the data. This is settled.", "labor economics"),
    ("Whatever. The idea that fossil fuels are still cheaper than renewables is outdated propaganda. Check the actual numbers.", "energy"),
    ("That's ridiculous. Claiming both sides are equally bad is intellectual laziness. False equivalence isn't analysis.", "politics"),
    ("Oh sure. Because deregulation always works out great. 2008 never happened, apparently. Selective memory much?", "financial regulation"),
    ("That's laughable. Anyone who thinks standardized tests measure learning doesn't understand education. They measure test-taking.", "education"),
    ("Come on now. The notion that private charity can replace social programs is ahistorical nonsense. It's been tried.", "social policy"),
    ("Whatever. Thinking authoritarian governments are more efficient is ignoring all of history. Read a book.", "political systems"),
    ("That's absurd. Anyone who denies systemic racism exists isn't paying attention to reality. The data is overwhelming.", "social justice"),
    ("Oh please. The idea that the market will solve climate change without intervention is fantasy. We don't have time.", "climate policy"),
    ("That's nonsense. Claiming vaccines are dangerous ignores mountains of evidence. This is anti-science.", "public health"),
    ("Come on. Anyone who thinks trickle-down benefits workers hasn't looked at wage stagnation. Wake up.", "economics"),
    ("Whatever. The notion that prison deters crime is contradicted by recidivism rates. This has been studied.", "criminal justice"),
    ("That's ridiculous. Thinking AI will solve all problems is techno-utopianism. It's a tool, not magic.", "technology"),
    ("Oh sure. Because giving everyone a gun makes us safer. That logic worked out great for America. Oh wait.", "gun policy"),
    ("That's laughable. Anyone who thinks cancel culture is the real problem needs perspective. Talk about first world problems.", "culture"),
    ("Come on now. The idea that bootstrapping works for everyone ignores structural barriers. This is privilege talking.", "opportunity"),
    ("Whatever. Claiming hydroxychloroquine works for COVID ignored the trials. Stop spreading misinformation.", "medicine"),
    ("That's absurd. Anyone who thinks corporations will self-regulate hasn't studied business history. They won't.", "regulation"),
    ("Oh please. The notion that both candidates are equally bad is lazy cynicism. Actually compare their records.", "politics")
]

for i, (text, topic) in enumerate(dismissive_samples, start=7):
    uncertainty, confidence = compute_scores(text)
    additional_tests.append({
        "id": f"test_dismissive_{i:03d}",
        "text": text,
        "topic": topic,
        "true_emotion": "dismissive",
        "true_emotion_idx": 3,
        "true_uncertainty": uncertainty,
        "true_confidence": confidence
    })

# === PASSIONATE (32 more: 006-037, skip test_passionate_007 which exists) ===
passionate_samples = [
    ("We absolutely must address wealth inequality! The gap between rich and poor threatens our democracy! This is the challenge that will define our generation!", "economic justice"),
    ("Criminal justice reform is absolutely essential for our society! Too many lives are destroyed by a broken system! We have to act now with courage!", "justice"),
    ("Access to clean water is a fundamental human right! Communities shouldn't have to choose between health and affordability! This matters so much!", "public health"),
    ("We must protect net neutrality! The free and open internet is vital for democracy and innovation! We cannot let corporations control access!", "internet freedom"),
    ("Public transit is absolutely crucial for livable cities! It reduces pollution, increases mobility, and creates community! We need massive investment!", "transportation"),
    ("Reproductive rights are fundamental to bodily autonomy! Everyone deserves to make their own healthcare decisions! This is about human dignity!", "healthcare rights"),
    ("We absolutely must reform our immigration system! These are human beings seeking better lives for their families! We have to show compassion!", "immigration"),
    ("Campaign finance reform is vital for democracy! Money is drowning out ordinary voices! We must ensure everyone has equal political power!", "election reform"),
    ("Affordable childcare is absolutely essential! Parents shouldn't have to choose between work and family! This is critical infrastructure!", "family policy"),
    ("We must protect labor rights! Workers deserve dignity, fair wages, and safe conditions! The labor movement built the middle class!", "labor"),
    ("Science funding is absolutely crucial for our future! Discovery and innovation drive progress! We must invest in research with commitment!", "science"),
    ("Food security is a fundamental right! No one should go hungry in a wealthy nation! We have the resources to feed everyone!", "hunger"),
    ("We absolutely must reform student loans! Debt is crushing an entire generation! Education should open doors, not trap people!", "education"),
    ("Rural broadband is vital for equality! Communities deserve access to opportunity! The digital divide is leaving people behind!", "technology access"),
    ("We must protect Social Security! Seniors deserve dignity after a lifetime of work! This is a sacred promise we must keep!", "retirement"),
    ("Paid family leave is absolutely essential! Parents deserve time to bond and care! This is about valuing families and children!", "family policy"),
    ("We absolutely must address homelessness! These are our neighbors who need help! Housing is a human right, not a privilege!", "housing"),
    ("Indigenous rights must be respected! Treaties are binding and lands are sacred! We have to honor these obligations!", "indigenous rights"),
    ("We must protect whistleblowers! They expose wrongdoing at great personal risk! Transparency is vital for accountability!", "government accountability"),
    ("Disability rights are absolutely fundamental! Everyone deserves full participation in society! Accessibility is justice, not charity!", "disability rights"),
    ("We absolutely must reform police training! Community safety requires trust and accountability! Lives are at stake!", "police reform"),
    ("Arts education is vital for development! Creativity and expression are essential skills! Every child deserves access to music and art!", "education"),
    ("We must protect endangered species! Biodiversity is irreplaceable! We're losing animals at an alarming rate and must act!", "conservation"),
    ("Living wages are absolutely essential! No one working full-time should live in poverty! Work must provide dignity!", "economic justice"),
    ("We absolutely must expand healthcare access! Health is a human right, not a commodity! No one should die from lack of care!", "healthcare"),
    ("Election security is vital for democracy! Every vote must count and be counted! We have to protect this sacred process!", "voting"),
    ("We must address food waste! Throwing away food while people go hungry is morally wrong! We can do so much better!", "sustainability"),
    ("Public libraries are absolutely vital! They provide access, community, and opportunity! We must fund them properly!", "public services"),
    ("We absolutely must protect journalists! Free press is essential for democracy! Attacks on reporters threaten everyone!", "press freedom"),
    ("Maternal healthcare is crucial! No one should die giving birth in a developed nation! This is preventable and unacceptable!", "healthcare"),
    ("We must combat gerrymandering! Districts should reflect communities, not partisan advantage! Democracy requires fair maps!", "voting rights"),
    ("Universal pre-K is absolutely essential! Early education shapes entire lives! Every child deserves this foundation!", "education")
]

# Note: skip test_passionate_006 to match the gap in existing data
for i, (text, topic) in enumerate(passionate_samples, start=6):
    if i == 7:  # Skip 007 since it already exists
        continue
    uncertainty, confidence = compute_scores(text)
    additional_tests.append({
        "id": f"test_passionate_{i:03d}",
        "text": text,
        "topic": topic,
        "true_emotion": "passionate",
        "true_emotion_idx": 4,
        "true_uncertainty": uncertainty,
        "true_confidence": confidence
    })

# === FRUSTRATED (32 more: 007-038) ===
frustrated_samples = [
    ("I've shown you the unemployment data three times now. Why are you still ignoring it? This is getting nowhere.", "economics"),
    ("How many times do I have to explain that sample size matters in statistics? Ugh. Are you even trying to understand?", "statistics"),
    ("For the fifth time, that's not how tariffs work. They're paid by importers, not exporters. Why is this so hard to grasp?", "trade policy"),
    ("I already explained that renewable energy can't work without storage. Are you not listening at all? This is exhausting.", "energy"),
    ("Seriously? We've been over this. The study you're citing was retracted. I told you that already. Pay attention.", "research"),
    ("How many sources do I need to provide? I've given you five peer-reviewed papers. What exactly would convince you?", "evidence"),
    ("This is the third time I've explained marginal tax rates. You're still confusing them with effective rates. Come on.", "taxation"),
    ("I've walked through the logic twice now. Where exactly are you getting confused? This shouldn't be this difficult.", "reasoning"),
    ("For the last time, correlation doesn't prove causation. How many times do we have to go over this? Basic research methods.", "methodology"),
    ("I already addressed that point ten minutes ago. Are you reading my responses or just waiting to talk?", "debate"),
    ("We've covered this twice already. That's not what the Constitution says. Actually read Article II. This is frustrating.", "law"),
    ("How is this still unclear? I've explained the difference between deficit and debt three times. These are basic terms.", "fiscal policy"),
    ("Seriously, that article you linked contradicts your point. Did you even read past the headline? This is absurd.", "media literacy"),
    ("I've shown you the CDC data twice. You keep citing blogs. Why are we not using actual sources?", "public health"),
    ("For the third time, inflation and interest rates aren't the same thing. How are you still confusing these?", "economics"),
    ("I already explained why that comparison doesn't work. They're completely different contexts. Are you listening?", "logic"),
    ("We've been through this. That's a strawman argument. I never said that. Stop misrepresenting my position.", "debate"),
    ("I've given you the historical examples twice now. You're ignoring all of them. What's the point of this discussion?", "history"),
    ("How many times - the study controlled for that variable. I told you this already. Read the methodology.", "research"),
    ("This is getting ridiculous. I've explained the difference between climate and weather four times now. Simple concepts.", "climate science"),
    ("For the last time, that's not how vaccines work. I've walked through the immunology twice. This is basic biology.", "medicine"),
    ("I already showed you why that analogy fails. Are you going to engage with the actual argument or keep deflecting?", "reasoning"),
    ("We've covered this ground three times. The data doesn't support your claim. I've shown you the numbers repeatedly.", "statistics"),
    ("How is this still confusing? GDP and GDP per capita are different metrics. I've explained this twice.", "economics"),
    ("Seriously? That's a logical fallacy. I've pointed out three times now. Can we have a rational discussion?", "logic"),
    ("I've provided five counterexamples to your claim. You've addressed none of them. This isn't productive.", "debate"),
    ("For the fourth time, anecdotes aren't data. Your personal experience doesn't override systematic studies.", "methodology"),
    ("I already explained why that source isn't credible. Check who funds it. We've been over this.", "source evaluation"),
    ("How many times - the experiment was double-blind. I explained the design twice. That addresses your concern.", "research"),
    ("This is exhausting. I've clarified my position three times and you're still attacking a version I don't hold.", "debate"),
    ("I've shown you the actual text of the law twice. You're arguing against something that isn't in there. Read it.", "policy"),
    ("For the last time, those numbers are from different years. You can't compare them directly. I've said this repeatedly.", "statistics")
]

for i, (text, topic) in enumerate(frustrated_samples, start=7):
    uncertainty, confidence = compute_scores(text)
    additional_tests.append({
        "id": f"test_frustrated_{i:03d}",
        "text": text,
        "topic": topic,
        "true_emotion": "frustrated",
        "true_emotion_idx": 5,
        "true_uncertainty": uncertainty,
        "true_confidence": confidence
    })

# === ANGRY (31 more: 007-037) ===
angry_samples = [
    ("That's an outrageous lie! The data clearly shows the opposite! I'm not going to sit here and let you spread this garbage!", "misinformation"),
    ("You're deliberately distorting the facts! This is dishonest and manipulative! I'm done with this bad-faith nonsense!", "debate"),
    ("That comparison is offensive and disgusting! How dare you equate those situations! This is morally repugnant!", "ethics"),
    ("You have zero understanding of this issue! Your ignorance is dangerous! Stop spreading this harmful misinformation!", "knowledge"),
    ("That's absolutely unacceptable! People's lives are at stake and you're playing political games! This is shameful!", "priorities"),
    ("You're completely ignoring the suffering this causes! That's callous and cruel! I refuse to engage with such heartlessness!", "empathy"),
    ("That policy would destroy communities! You clearly don't care about the real-world impact! This is unconscionable!", "policy"),
    ("Your position is morally bankrupt! You're prioritizing profit over people's lives! This makes me furious!", "values"),
    ("That's a disgusting characterization! You're demonizing vulnerable people! This rhetoric is dangerous and wrong!", "framing"),
    ("You're spreading dangerous conspiracy theories! This gets people hurt! I'm appalled that you'd platform this!", "misinformation"),
    ("That's victim-blaming at its worst! How dare you suggest they brought this on themselves! This is reprehensible!", "accountability"),
    ("You're defending the indefensible! There's no justification for that behavior! This is morally abhorrent!", "ethics"),
    ("That's pure propaganda! You're parroting talking points instead of thinking critically! This is intellectual cowardice!", "reasoning"),
    ("You're trivializing real harm! People are suffering and you're making jokes! That's disgusting!", "seriousness"),
    ("That interpretation is completely dishonest! You're twisting words to fit your agenda! This is manipulative!", "framing"),
    ("You have no right to speak for that community! This is appropriation and it's wrong! Stay in your lane!", "representation"),
    ("That's deeply offensive! You're perpetuating harmful stereotypes! I can't believe you'd say something so ignorant!", "prejudice"),
    ("You're enabling abuse with that position! That makes you complicit! This is unacceptable!", "accountability"),
    ("That's historical revisionism! You're whitewashing atrocities! This is disrespectful to victims and their families!", "history"),
    ("You're punching down at vulnerable people! That's cruel and cowardly! Pick on someone your own size!", "power dynamics"),
    ("That policy would kill people! You're valuing money over human lives! This is evil!", "priorities"),
    ("You're denying people's lived experiences! That's gaslighting! Show some basic respect!", "validation"),
    ("That's textbook discrimination! You're defending bigotry! This is morally wrong!", "justice"),
    ("You're spreading harmful medical misinformation! This is dangerous! People could die because of this!", "public health"),
    ("That comparison minimizes genocide! How dare you! This is offensive to survivors and historians!", "history"),
    ("You're defending corruption! That money was stolen from taxpayers! This should outrage everyone!", "governance"),
    ("That's environmental destruction for profit! You're selling out future generations! This is shortsighted and wrong!", "environment"),
    ("You're blaming the poor for their poverty! That's classist garbage! Check your privilege!", "economic justice"),
    ("That's defending police brutality! People died! How can you justify this violence!", "justice"),
    ("You're dismissing science because it's inconvenient! That's willful ignorance! This is dangerous!", "science"),
    ("That's corporate apologism! You're defending exploitation! Workers deserve so much better than this!", "labor")
]

for i, (text, topic) in enumerate(angry_samples, start=7):
    uncertainty, confidence = compute_scores(text)
    additional_tests.append({
        "id": f"test_angry_{i:03d}",
        "text": text,
        "topic": topic,
        "true_emotion": "angry",
        "true_emotion_idx": 6,
        "true_uncertainty": uncertainty,
        "true_confidence": confidence
    })

# === SARCASTIC (31 more: 007-037) ===
sarcastic_samples = [
    ("Oh wonderful, another libertarian explaining how roads would totally work without government. Fascinating theory. Very practical.", "political theory"),
    ("Yeah, I'm sure thoughts and prayers will fix infrastructure. That's definitely how engineering works. Great plan.", "policy"),
    ("Oh absolutely, because one anecdote from your cousin totally disproves peer-reviewed research. Solid methodology there.", "evidence"),
    ("Sure, let's just ignore centuries of economic history. I'm sure it's different this time. What could go wrong?", "economics"),
    ("Oh yes, because YouTube videos are definitely equivalent to medical degrees. Such reliable expertise.", "credibility"),
    ("Right, the real problem is people being too sensitive, not actual discrimination. Hot take. Very brave.", "social issues"),
    ("Oh fantastic, another 'both sides' argument. Because nuance means everything is exactly the same. Brilliant.", "politics"),
    ("Yeah, regulations are why businesses fail, not bad management. That's definitely the issue. Keep telling yourself that.", "business"),
    ("Oh sure, because this time deregulation will work differently. Unlike every other time. Magic!", "policy"),
    ("Right, scientists are all conspiring for that sweet grant money. Makes perfect sense. Very logical.", "science"),
    ("Oh absolutely, because a CEO definitely works 300 times harder than their workers. Simple math.", "labor economics"),
    ("Yeah, the free market will solve climate change any day now. Just wait. It's coming. Any minute.", "climate"),
    ("Oh wonderful, another slippery slope argument. Because those are always so well-reasoned. Compelling stuff.", "logic"),
    ("Sure, let's base national policy on what feels true to you. Data is overrated anyway. Go with your gut.", "policy making"),
    ("Oh yes, because corporations definitely have consumers' best interests at heart. That's how capitalism works. Obviously.", "business"),
    ("Right, the problem with education is it's too accessible. That's definitely the issue. Nailed it.", "education"),
    ("Oh absolutely, because one exception totally disproves the statistical trend. You've cracked the code.", "statistics"),
    ("Yeah, I'm sure this opinion piece from that totally unbiased source settles the science. Case closed.", "evidence"),
    ("Oh fantastic, another 'do your own research' comment. Because Google equals expertise now. Good to know.", "research"),
    ("Sure, the media is completely making up climate change for... reasons. That conspiracy makes total sense.", "media"),
    ("Oh yes, because reducing complex policy to a bumper sticker slogan is definitely insightful. Deep thinking.", "politics"),
    ("Right, the real discrimination is against the majority group. That's definitely how power works. Sure.", "social justice"),
    ("Oh absolutely, because if we're nice to corporations they'll definitely share the wealth. Any day now.", "economics"),
    ("Yeah, one bad experience totally means the entire field is fraudulent. Sound reasoning. Very scientific.", "methodology"),
    ("Oh wonderful, another 'common sense' argument that ignores actual expertise. Why study anything, really?", "knowledge"),
    ("Sure, let's ignore what works everywhere else because we're special. American exceptionalism at its finest.", "policy"),
    ("Oh yes, because the founding fathers definitely anticipated AI and social media. Very applicable.", "law"),
    ("Right, poor people should just stop being poor. Wow. Why didn't they think of that? Genius.", "poverty"),
    ("Oh absolutely, because taxation is exactly like armed robbery. Perfect analogy. No flaws there.", "taxation"),
    ("Yeah, I'm sure eliminating all regulations would create paradise, not chaos. Definitely. No question.", "regulation"),
    ("Oh fantastic, another appeal to nature fallacy. Because everything natural is good. Like arsenic.", "logic")
]

for i, (text, topic) in enumerate(sarcastic_samples, start=7):
    uncertainty, confidence = compute_scores(text)
    additional_tests.append({
        "id": f"test_sarcastic_{i:03d}",
        "text": text,
        "topic": topic,
        "true_emotion": "sarcastic",
        "true_emotion_idx": 7,
        "true_uncertainty": uncertainty,
        "true_confidence": confidence
    })

# Load existing test data
test_data_path = Path("test_data/test_set.json")
with open(test_data_path) as f:
    existing_tests = json.load(f)

# Combine and save
all_tests = existing_tests + additional_tests
with open(test_data_path, 'w') as f:
    json.dump(all_tests, f, indent=2)

print(f"✅ Generated {len(additional_tests)} new test cases")
print(f"   Total test set size: {len(all_tests)}")
print(f"\n📊 Distribution:")
emotions = ["calm", "confident", "defensive", "dismissive", "passionate", "frustrated", "angry", "sarcastic"]
for emotion in emotions:
    count = sum(1 for t in all_tests if t['true_emotion'] == emotion)
    print(f"   {emotion:12}: {count} samples")
