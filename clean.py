"""
Clean Recommendation Data

This script loads the original recommendation word file and filters out
non-informative or duplicate content. It extracts relevant sections and
removes entries that are empty or too short, to prepare a cleaner dataset
for downstream tasks like knowledge base building or vectorization.
"""

import re
import json
import logging
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from docx import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define sections to extract from the document
SECTIONS = [
    "Architecture",
    "Communications",
    "Cache",
    "Programming languages and execution environment",
    "Data",
    "Development, test and integration tools",
    "Operations and End of Life",
]

# Comprehensive list of recommendation indicators
RECOMMENDATION_INDICATORS = {
    # Direct recommendation phrases
    "explicit_recommendations": [
        "we recommend", "recommendation:", "recommended", "our recommendation", 
        "it is recommended", "is recommended", "are recommended", "recommendation is",
        "recommendations include", "recommended approach", "recommended practice",
        "recommended solution", "recommended strategy", "recommended method",
        "recommended tool", "recommended framework", "recommended library",
        "recommended configuration", "recommended setting", "recommended option",
        "recommended value", "recommended parameter", "recommended architecture",
        "recommended design", "recommended pattern", "recommended implementation",
        "recommended way", "recommended to", "recommend that", "recommend to",
        "recommend using", "recommend adopting", "recommend implementing",
        "recommend considering", "recommend choosing", "recommend selecting",
        "recommend following", "recommend avoiding", "recommend against",
        "strongly recommend", "highly recommend", "would recommend",
        "we advise", "advise that", "advise to", "advised to", "advisable to",
        "we suggest", "suggest that", "suggest to", "suggested to", "suggestion is",
        "our suggestion", "suggestions include", "we propose", "propose that", "proposal is",
        "our proposal", "we advocate", "advocate for", "advocating for", "we endorse",
        "endorse the", "endorsement of", "we favor", "favor the", "favorable approach",
        "we support", "support the", "supporting the", "we back", "back the", "backing the",
        "we encourage", "encourage the", "encouraging the", "we promote", "promote the",
        "promoting the", "we urge", "urge that", "urging the", "we counsel", "counsel that",
        "counseling to", "we guide", "guide towards", "guidance is", "our guidance",
        "we direct", "direct that", "directing to", "we prescribe", "prescribe that",
        "prescribing to", "we instruct", "instruct to", "instruction is", "our instruction",
        "we advise against", "advise against", "advised against", "we discourage",
        "discourage the", "discouraging the", "we caution against", "caution against",
        "cautioning against", "we warn against", "warn against", "warning against",
        "we suggest avoiding", "suggest avoiding", "suggested avoiding",
        "we recommend against", "recommend against", "recommended against",
        "best recommendation", "key recommendation", "primary recommendation",
        "critical recommendation", "important recommendation", "essential recommendation",
        "valuable recommendation", "practical recommendation", "useful recommendation",
        "effective recommendation", "efficient recommendation", "strategic recommendation",
        "tactical recommendation", "technical recommendation", "architectural recommendation",
        "design recommendation", "implementation recommendation", "operational recommendation",
        "development recommendation", "security recommendation", "performance recommendation",
        "scalability recommendation", "reliability recommendation", "maintainability recommendation",
        "usability recommendation", "accessibility recommendation", "compatibility recommendation",
        "interoperability recommendation", "portability recommendation", "extensibility recommendation",
        "modularity recommendation", "reusability recommendation", "testability recommendation",
        "deployability recommendation", "monitorability recommendation", "observability recommendation",
        "our advice is", "advice is to", "advising to", "advised approach", "advised method",
        "advised strategy", "advised solution", "advised practice", "advised technique",
        "advised implementation", "advised configuration", "advised setup", "advised architecture",
        "advised design", "advised pattern", "advised framework", "advised tool", "advised library",
        "advised component", "advised system", "advised platform", "advised technology",
        "advised language", "advised protocol", "advised standard", "advised convention",
        "advised guideline", "advised principle", "advised rule", "advised policy",
        "advised procedure", "advised process", "advised workflow", "advised lifecycle",
        "advised methodology", "advised approach", "advised paradigm", "advised model",
        "advised structure", "advised organization", "advised arrangement", "advised composition"
    ],
    
    # Imperative verbs (often start recommendations)
    "imperative_verbs": [
        "use ", "avoid ", "implement ", "consider ", "choose ", "select ",
        "prefer ", "adopt ", "follow ", "maintain ", "create ", "develop ",
        "establish ", "ensure ", "prevent ", "limit ", "maximize ", "minimize ",
        "optimize ", "standardize ", "leverage ", "utilize ", "employ ",
        "incorporate ", "integrate ", "apply ", "deploy ", "configure ",
        "set up ", "install ", "build ", "design ", "architect ", "structure ",
        "organize ", "define ", "specify ", "document ", "test ", "validate ",
        "verify ", "monitor ", "track ", "log ", "analyze ", "evaluate ",
        "assess ", "review ", "refactor ", "update ", "upgrade ", "migrate ",
        "transition ", "move to ", "switch to ", "change to ", "convert to ",
        "replace ", "substitute ", "eliminate ", "remove ", "disable ", "enable ",
        "activate ", "deactivate ", "start ", "stop ", "pause ", "resume ",
        "continue ", "discontinue ", "proceed with ", "halt ", "terminate ",
        "end ", "begin ", "initiate ", "launch ", "execute ", "run ",
        "perform ", "conduct ", "carry out ", "undertake ", "accomplish ",
        "achieve ", "attain ", "realize ", "fulfill ", "complete ", "finish ",
        "conclude ", "wrap up ", "wind down ", "phase out ", "introduce ",
        "present ", "offer ", "provide ", "supply ", "furnish ", "deliver ",
        "produce ", "generate ", "create ", "construct ", "assemble ", "compose ",
        "formulate ", "devise ", "invent ", "innovate ", "pioneer ", "spearhead ",
        "lead ", "guide ", "direct ", "steer ", "navigate ", "pilot ", "drive ",
        "manage ", "administer ", "oversee ", "supervise ", "control ", "govern ",
        "regulate ", "adjust ", "tune ", "calibrate ", "align ", "synchronize ",
        "harmonize ", "balance ", "equalize ", "stabilize ", "secure ", "protect ",
        "safeguard ", "shield ", "defend ", "guard ", "preserve ", "conserve ",
        "retain ", "keep ", "maintain ", "sustain ", "support ", "uphold ",
        "reinforce ", "strengthen ", "enhance ", "augment ", "amplify ", "boost ",
        "increase ", "decrease ", "reduce ", "lower ", "diminish ", "lessen ",
        "shrink ", "contract ", "expand ", "extend ", "enlarge ", "broaden ",
        "widen ", "deepen ", "heighten ", "elevate ", "raise ", "lift ", "hoist ",
        "accelerate ", "expedite ", "hasten ", "quicken ", "speed up ", "slow down ",
        "delay ", "postpone ", "defer ", "schedule ", "plan ", "arrange ", "prepare ",
        "ready ", "equip ", "outfit ", "furnish ", "provision ", "stock ", "supply ",
        "allocate ", "assign ", "distribute ", "divide ", "partition ", "segment ",
        "separate ", "isolate ", "segregate ", "detach ", "disconnect ", "disengage ",
        "unlink ", "decouple ", "connect ", "link ", "join ", "unite ", "combine ",
        "merge ", "consolidate ", "amalgamate ", "fuse ", "blend ", "mix ", "intermingle ",
        "intertwine ", "interweave ", "interlace ", "interlock ", "interrelate ",
        "correlate ", "associate ", "dissociate ", "disassociate ", "disassemble ",
        "dismantle ", "take apart ", "break down ", "decompose ", "deconstruct "
    ],
    
    # Modal verbs and obligation phrases
    "obligation_phrases": [
        "should ", "must ", "need to ", "have to ", "ought to ", "required to ",
        "necessary to ", "essential to ", "critical to ", "important to ",
        "crucial to ", "vital to ", "significant to ", "mandatory to ",
        "compulsory to ", "obligatory to ", "should not ", "must not ",
        "do not ", "don't ", "never ", "always ", "shall ", "shall not ",
        "expected to ", "supposed to ", "advised to ", "encouraged to ",
        "discouraged from ", "prohibited from ", "forbidden to ", "restricted from ",
        "it is necessary ", "it is essential ", "it is critical ", "it is important ",
        "it is crucial ", "it is vital ", "it is significant ", "it is mandatory ",
        "it is compulsory ", "it is obligatory ", "it is required ", "it is expected ",
        "it is advised ", "it is encouraged ", "it is recommended ", "it is suggested ",
        "it is proposed ", "it is advocated ", "it is endorsed ", "it is supported ",
        "it is favored ", "it is preferred ", "it is desirable ", "it is beneficial ",
        "it is advantageous ", "it is profitable ", "it is rewarding ", "it is worthwhile ",
        "it is valuable ", "it is useful ", "it is helpful ", "it is practical ",
        "it is pragmatic ", "it is sensible ", "it is reasonable ", "it is logical ",
        "it is rational ", "it is sound ", "it is wise ", "it is prudent ",
        "it is judicious ", "it is advisable ", "it is appropriate ", "it is suitable ",
        "it is fitting ", "it is proper ", "it is correct ", "it is right ",
        "it is imperative ", "it is paramount ", "it is indispensable ", "it is unavoidable ",
        "it is inescapable ", "it is inevitable ", "it is needed ", "it is demanded ",
        "it is called for ", "it is warranted ", "it is justified ", "it is merited ",
        "it is deserved ", "it is earned ", "it is due ", "it is owed ",
        "it is incumbent ", "it is binding ", "it is compelling ", "it is urgent ",
        "it is pressing ", "it is immediate ", "it is instant ", "it is prompt ",
        "it is expedient ", "it is expeditious ", "it is swift ", "it is quick ",
        "it is rapid ", "it is fast ", "it is speedy ", "it is hasty ",
        "it is hurried ", "it is rushed ", "it is accelerated ", "it is expedited ",
        "it is forbidden ", "it is prohibited ", "it is banned ", "it is barred ",
        "it is disallowed ", "it is proscribed ", "it is vetoed ", "it is rejected ",
        "it is refused ", "it is denied ", "it is disapproved ", "it is discouraged ",
        "it is frowned upon ", "it is looked down upon ", "it is condemned ", "it is denounced ",
        "it is criticized ", "it is censured ", "it is reproached ", "it is rebuked ",
        "it is reprimanded ", "it is admonished ", "it is chastised ", "it is castigated ",
        "it is upbraided ", "it is scolded ", "it is chided ", "it is berated ",
        "it is required that ", "it is necessary that ", "it is essential that ",
        "it is critical that ", "it is important that ", "it is crucial that ",
        "it is vital that ", "it is significant that ", "it is mandatory that ",
        "it is compulsory that ", "it is obligatory that ", "it is expected that ",
        "it is advised that ", "it is encouraged that ", "it is recommended that ",
        "it is suggested that ", "it is proposed that ", "it is advocated that ",
        "it is endorsed that ", "it is supported that ", "it is favored that ",
        "it is preferred that ", "it is desirable that ", "it is beneficial that ",
        "it is advantageous that ", "it is profitable that ", "it is rewarding that ",
        "it is worthwhile that ", "it is valuable that ", "it is useful that ",
        "it is helpful that ", "it is practical that ", "it is pragmatic that ",
        "it is sensible that ", "it is reasonable that ", "it is logical that ",
        "it is rational that ", "it is sound that ", "it is wise that ",
        "it is prudent that ", "it is judicious that ", "it is advisable that ",
        "it is appropriate that ", "it is suitable that ", "it is fitting that ",
        "it is proper that ", "it is correct that ", "it is right that ",
        "it is imperative that ", "it is paramount that ", "it is indispensable that ",
        "it is unavoidable that ", "it is inescapable that ", "it is inevitable that ",
        "it is needed that ", "it is demanded that ", "it is called for that ",
        "it is warranted that ", "it is justified that ", "it is merited that ",
        "it is deserved that ", "it is earned that ", "it is due that ",
        "it is owed that ", "it is incumbent that ", "it is binding that ",
        "it is compelling that ", "it is urgent that ", "it is pressing that ",
        "it is immediate that ", "it is instant that ", "it is prompt that ",
        "it is expedient that ", "it is expeditious that ", "it is swift that ",
        "it is quick that ", "it is rapid that ", "it is fast that ",
        "it is speedy that ", "it is hasty that ", "it is hurried that ",
        "it is rushed that ", "it is accelerated that ", "it is expedited that "
    ],
    
      
    # Best practice phrases
"best_practice_phrases": [
    # Core best practice phrases
    "best practice", "good practice", "standard practice", "common practice",
    "industry practice", "industry standard", "standard approach", "best approach",
    "preferred approach", "optimal approach", "ideal approach", "effective approach",
    "efficient approach", "proven approach", "established approach", "conventional approach",
    "typical approach", "accepted approach", "recognized approach", "recommended approach",
    "suggested approach", "advisable approach", "sensible approach", "practical approach",
    "pragmatic approach", "strategic approach", "tactical approach", "methodical approach",
    "systematic approach", "structured approach", "organized approach", "disciplined approach",
    
    # Method variations
    "best method", "good method", "standard method", "common method", "industry method",
    "preferred method", "optimal method", "ideal method", "effective method", "efficient method",
    "proven method", "established method", "conventional method", "typical method",
    "accepted method", "recognized method", "recommended method", "suggested method",
    "advisable method", "sensible method", "practical method", "pragmatic method",
    "strategic method", "tactical method", "methodical method", "systematic method",
    "structured method", "organized method", "disciplined method",
    
    # Technique variations
    "best technique", "good technique", "standard technique", "common technique", "industry technique",
    "preferred technique", "optimal technique", "ideal technique", "effective technique",
    "efficient technique", "proven technique", "established technique", "conventional technique",
    "typical technique", "accepted technique", "recognized technique", "recommended technique",
    "suggested technique", "advisable technique", "sensible technique", "practical technique",
    "pragmatic technique", "strategic technique", "tactical technique", "methodical technique",
    "systematic technique", "structured technique", "organized technique", "disciplined technique",
    
    # Strategy variations
    "best strategy", "good strategy", "standard strategy", "common strategy", "industry strategy",
    "preferred strategy", "optimal strategy", "ideal strategy", "effective strategy",
    "efficient strategy", "proven strategy", "established strategy", "conventional strategy",
    "typical strategy", "accepted strategy", "recognized strategy", "recommended strategy",
    "suggested strategy", "advisable strategy", "sensible strategy", "practical strategy",
    "pragmatic strategy", "strategic strategy", "tactical strategy", "methodical strategy",
    "systematic strategy", "structured strategy", "organized strategy", "disciplined strategy",
    
    # Solution variations
    "best solution", "good solution", "standard solution", "common solution", "industry solution",
    "preferred solution", "optimal solution", "ideal solution", "effective solution",
    "efficient solution", "proven solution", "established solution", "conventional solution",
    "typical solution", "accepted solution", "recognized solution", "recommended solution",
    "suggested solution", "advisable solution", "sensible solution", "practical solution",
    "pragmatic solution", "strategic solution", "tactical solution", "methodical solution",
    "systematic solution", "structured solution", "organized solution", "disciplined solution",
    
    # Pattern variations
    "best pattern", "good pattern", "standard pattern", "common pattern", "industry pattern",
    "preferred pattern", "optimal pattern", "ideal pattern", "effective pattern",
    "efficient pattern", "proven pattern", "established pattern", "conventional pattern",
    "typical pattern", "accepted pattern", "recognized pattern", "recommended pattern",
    "suggested pattern", "advisable pattern", "sensible pattern", "practical pattern",
    "pragmatic pattern", "strategic pattern", "tactical pattern", "methodical pattern",
    "systematic pattern", "structured pattern", "organized pattern", "disciplined pattern",
    
    # Design variations
    "best design", "good design", "standard design", "common design", "industry design",
    "preferred design", "optimal design", "ideal design", "effective design",
    "efficient design", "proven design", "established design", "conventional design",
    "typical design", "accepted design", "recognized design", "recommended design",
    "suggested design", "advisable design", "sensible design", "practical design",
    "pragmatic design", "strategic design", "tactical design", "methodical design",
    "systematic design", "structured design", "organized design", "disciplined design",
    
    # Architecture variations
    "best architecture", "good architecture", "standard architecture", "common architecture", 
    "industry architecture", "preferred architecture", "optimal architecture", "ideal architecture", 
    "effective architecture", "efficient architecture", "proven architecture", "established architecture", 
    "conventional architecture", "typical architecture", "accepted architecture", "recognized architecture", 
    "recommended architecture", "suggested architecture", "advisable architecture", "sensible architecture", 
    "practical architecture", "pragmatic architecture", "strategic architecture", "tactical architecture", 
    "methodical architecture", "systematic architecture", "structured architecture", "organized architecture", 
    "disciplined architecture",
    
    # Implementation variations
    "best implementation", "good implementation", "standard implementation", "common implementation", 
    "industry implementation", "preferred implementation", "optimal implementation", "ideal implementation", 
    "effective implementation", "efficient implementation", "proven implementation", "established implementation", 
    "conventional implementation", "typical implementation", "accepted implementation", "recognized implementation", 
    "recommended implementation", "suggested implementation", "advisable implementation", "sensible implementation", 
    "practical implementation", "pragmatic implementation", "strategic implementation", "tactical implementation", 
    "methodical implementation", "systematic implementation", "structured implementation", "organized implementation", 
    "disciplined implementation",
    
    # Configuration variations
    "best configuration", "good configuration", "standard configuration", "common configuration", 
    "industry configuration", "preferred configuration", "optimal configuration", "ideal configuration", 
    "effective configuration", "efficient configuration", "proven configuration", "established configuration", 
    "conventional configuration", "typical configuration", "accepted configuration", "recognized configuration", 
    "recommended configuration", "suggested configuration", "advisable configuration", "sensible configuration", 
    "practical configuration", "pragmatic configuration", "strategic configuration", "tactical configuration", 
    "methodical configuration", "systematic configuration", "structured configuration", "organized configuration", 
    "disciplined configuration",
    
    # Setup variations
    "best setup", "good setup", "standard setup", "common setup", "industry setup",
    "preferred setup", "optimal setup", "ideal setup", "effective setup",
    "efficient setup", "proven setup", "established setup", "conventional setup",
    "typical setup", "accepted setup", "recognized setup", "recommended setup",
    "suggested setup", "advisable setup", "sensible setup", "practical setup",
    "pragmatic setup", "strategic setup", "tactical setup", "methodical setup",
    "systematic setup", "structured setup", "organized setup", "disciplined setup",
    
    # Guideline variations
    "best guideline", "good guideline", "standard guideline", "common guideline", "industry guideline",
    "preferred guideline", "optimal guideline", "ideal guideline", "effective guideline",
    "efficient guideline", "proven guideline", "established guideline", "conventional guideline",
    "typical guideline", "accepted guideline", "recognized guideline", "recommended guideline",
    "suggested guideline", "advisable guideline", "sensible guideline", "practical guideline",
    "pragmatic guideline", "strategic guideline", "tactical guideline", "methodical guideline",
    "systematic guideline", "structured guideline", "organized guideline", "disciplined guideline",
    
    # Principle variations
    "best principle", "good principle", "standard principle", "common principle", "industry principle",
    "preferred principle", "optimal principle", "ideal principle", "effective principle",
    "efficient principle", "proven principle", "established principle", "conventional principle",
    "typical principle", "accepted principle", "recognized principle", "recommended principle",
    "suggested principle", "advisable principle", "sensible principle", "practical principle",
    "pragmatic principle", "strategic principle", "tactical principle", "methodical principle",
    "systematic principle", "structured principle", "organized principle", "disciplined principle",
    
    # Convention variations
    "best convention", "good convention", "standard convention", "common convention", "industry convention",
    "preferred convention", "optimal convention", "ideal convention", "effective convention",
    "efficient convention", "proven convention", "established convention", "conventional convention",
    "typical convention", "accepted convention", "recognized convention", "recommended convention",
    "suggested convention", "advisable convention", "sensible convention", "practical convention",
    "pragmatic convention", "strategic convention", "tactical convention", "methodical convention",
    "systematic convention", "structured convention", "organized convention", "disciplined convention",
    
    # Phrases with "way"
    "best way", "good way", "standard way", "common way", "industry way",
    "preferred way", "optimal way", "ideal way", "effective way",
    "efficient way", "proven way", "established way", "conventional way",
    "typical way", "accepted way", "recognized way", "recommended way",
    "suggested way", "advisable way", "sensible way", "practical way",
    "pragmatic way", "strategic way", "tactical way", "methodical way",
    "systematic way", "structured way", "organized way", "disciplined way",
    
    # Phrases with "manner"
    "best manner", "good manner", "standard manner", "common manner", "industry manner",
    "preferred manner", "optimal manner", "ideal manner", "effective manner",
    "efficient manner", "proven manner", "established manner", "conventional manner",
    "typical manner", "accepted manner", "recognized manner", "recommended manner",
    "suggested manner", "advisable manner", "sensible manner", "practical manner",
    "pragmatic manner", "strategic manner", "tactical manner", "methodical manner",
    "systematic manner", "structured manner", "organized manner", "disciplined manner",
    
    # Phrases with "procedure"
    "best procedure", "good procedure", "standard procedure", "common procedure", "industry procedure",
    "preferred procedure", "optimal procedure", "ideal procedure", "effective procedure",
    "efficient procedure", "proven procedure", "established procedure", "conventional procedure",
    "typical procedure", "accepted procedure", "recognized procedure", "recommended procedure",
    "suggested procedure", "advisable procedure", "sensible procedure", "practical procedure",
    "pragmatic procedure", "strategic procedure", "tactical procedure", "methodical procedure",
    "systematic procedure", "structured procedure", "organized procedure", "disciplined procedure"
],

# Preference and comparison phrases
"preference_phrases": [
    # Basic preference phrases
    "preferable to", "better than", "preferred over", "rather than", "instead of",
    "superior to", "more effective than", "more efficient than", "more reliable than",
    "more secure than", "more robust than", "more scalable than", "more maintainable than",
    "more performant than", "more suitable than", "more appropriate than", "more applicable than",
    "advantageous over", "benefits over", "improvements over", "enhancements over",
    "preferred choice", "better choice", "optimal choice", "ideal choice", "best choice",
    "preferred option", "better option", "optimal option", "ideal option", "best option",
    "preferred solution", "better solution", "optimal solution", "ideal solution", "best solution",
    
    # Additional comparative phrases
    "more optimal than", "more ideal than", "more preferable than", "more advantageous than",
    "more beneficial than", "more favorable than", "more desirable than", "more practical than",
    "more pragmatic than", "more sensible than", "more reasonable than", "more logical than",
    "more rational than", "more sound than", "more wise than", "more prudent than",
    "more judicious than", "more advisable than", "more appropriate than", "more suitable than",
    "more fitting than", "more proper than", "more correct than", "more right than",
    "more accurate than", "more precise than", "more exact than", "more specific than",
    "more targeted than", "more focused than", "more directed than", "more oriented than",
    "more aligned than", "more consistent than", "more coherent than", "more cohesive than",
    "more unified than", "more integrated than", "more harmonious than", "more balanced than",
    "more proportional than", "more symmetrical than", "more even than", "more uniform than",
    "more regular than", "more standardized than", "more normalized than", "more formalized than",
    "more structured than", "more organized than", "more systematic than", "more methodical than",
    "more orderly than", "more disciplined than", "more controlled than", "more managed than",
    "more supervised than", "more regulated than", "more governed than", "more administered than",
    "more directed than", "more guided than", "more steered than", "more navigated than",
    "more piloted than", "more driven than", "more led than", "more conducted than",
    "more executed than", "more performed than", "more carried out than", "more implemented than",
    "more applied than", "more utilized than", "more employed than", "more used than",
    "more operated than", "more handled than", "more manipulated than", "more maneuvered than",
    "more worked than", "more processed than", "more treated than", "more dealt with than",
    
    # Preference phrases with "recommend"
    "recommend over", "recommend instead of", "recommend rather than", "recommend in place of",
    "recommend in preference to", "recommend as an alternative to", "recommend as a substitute for",
    "recommend as a replacement for", "recommend as a successor to", "recommend as an improvement over",
    "recommend as an enhancement to", "recommend as an upgrade from", "recommend as a better option than",
    "recommend as a superior choice to", "recommend as a preferable selection to",
    
    # Preference phrases with "suggest"
    "suggest over", "suggest instead of", "suggest rather than", "suggest in place of",
    "suggest in preference to", "suggest as an alternative to", "suggest as a substitute for",
    "suggest as a replacement for", "suggest as a successor to", "suggest as an improvement over",
    "suggest as an enhancement to", "suggest as an upgrade from", "suggest as a better option than",
    "suggest as a superior choice to", "suggest as a preferable selection to",
    
    # Preference phrases with "advise"
    "advise over", "advise instead of", "advise rather than", "advise in place of",
    "advise in preference to", "advise as an alternative to", "advise as a substitute for",
    "advise as a replacement for", "advise as a successor to", "advise as an improvement over",
    "advise as an enhancement to", "advise as an upgrade from", "advise as a better option than",
    "advise as a superior choice to", "advise as a preferable selection to",
    
    # Preference phrases with "favor"
    "favor over", "favor instead of", "favor rather than", "favor in place of",
    "favor in preference to", "favor as an alternative to", "favor as a substitute for",
    "favor as a replacement for", "favor as a successor to", "favor as an improvement over",
    "favor as an enhancement to", "favor as an upgrade from", "favor as a better option than",
    "favor as a superior choice to", "favor as a preferable selection to",
    
    # Preference phrases with "prefer"
    "prefer over", "prefer instead of", "prefer rather than", "prefer in place of",
    "prefer in preference to", "prefer as an alternative to", "prefer as a substitute for",
    "prefer as a replacement for", "prefer as a successor to", "prefer as an improvement over",
    "prefer as an enhancement to", "prefer as an upgrade from", "prefer as a better option than",
    "prefer as a superior choice to", "prefer as a preferable selection to",
    
    # Preference phrases with "choose"
    "choose over", "choose instead of", "choose rather than", "choose in place of",
    "choose in preference to", "choose as an alternative to", "choose as a substitute for",
    "choose as a replacement for", "choose as a successor to", "choose as an improvement over",
    "choose as an enhancement to", "choose as an upgrade from", "choose as a better option than",
    "choose as a superior choice to", "choose as a preferable selection to",
    
    # Preference phrases with "select"
    "select over", "select instead of", "select rather than", "select in place of",
    "select in preference to", "select as an alternative to", "select as a substitute for",
    "select as a replacement for", "select as a successor to", "select as an improvement over",
    "select as an enhancement to", "select as an upgrade from", "select as a better option than",
    "select as a superior choice to", "select as a preferable selection to", "select as a more effective solution than",
    "select as a more efficient approach than", "select as a more reliable method than",
    "select as a more secure implementation than", "select as a more robust architecture than",
    "select as a more scalable design than", "select as a more maintainable codebase than",
    "select as a more performant system than", "select as a more suitable technology than",
    "select as a more appropriate framework than", "select as a more applicable tool than",
    "select as a more advantageous platform than", "select as a more beneficial service than",
    "select as a more favorable component than", "select as a more desirable product than",
    "select as a more practical utility than", "select as a more pragmatic resource than",
    "select as a more sensible choice than", "select as a more reasonable alternative than",
    "select as a more logical option than", "select as a more rational selection than",
    "select as a more sound decision than", "select as a more wise investment than",
    "select as a more prudent direction than", "select as a more judicious path than"
],
    
    # Consequence phrases (often indicate recommendations)
"consequence_phrases": [
    # Basic consequence phrases
    "to ensure", "to prevent", "to avoid", "to reduce", "to minimize", "to maximize",
    "to optimize", "to improve", "to enhance", "to increase", "to decrease", "to maintain",
    "to achieve", "to enable", "to facilitate", "to support", "to promote", "to encourage",
    "to discourage", "to mitigate", "to address", "to resolve", "to overcome", "to handle",
    "to manage", "to control", "to regulate", "to streamline", "to simplify", "to accelerate",
    "in order to", "so that", "such that", "thereby", "thus ensuring", "thus preventing",
    "which ensures", "which prevents", "which avoids", "which reduces", "which improves",
    
    # Additional consequence phrases
    "to guarantee", "to secure", "to safeguard", "to protect", "to shield", "to defend",
    "to guard", "to preserve", "to conserve", "to retain", "to keep", "to sustain",
    "to uphold", "to reinforce", "to strengthen", "to augment", "to amplify", "to boost",
    "to elevate", "to raise", "to lift", "to heighten", "to deepen", "to widen",
    "to broaden", "to extend", "to expand", "to enlarge", "to grow", "to develop",
    "to advance", "to progress", "to further", "to forward", "to promote", "to foster",
    "to nurture", "to cultivate", "to nourish", "to feed", "to fuel", "to power",
    "to energize", "to invigorate", "to revitalize", "to rejuvenate", "to renew", "to refresh",
    "to restore", "to revive", "to rekindle", "to reignite", "to reactivate", "to reestablish",
    "to reinstate", "to reintroduce", "to reimplement", "to redeploy", "to repurpose", "to reconfigure",
    "to restructure", "to reorganize", "to rearrange", "to realign", "to readjust", "to recalibrate",
    "to rebalance", "to redistribute", "to reallocate", "to reassign", "to reposition", "to relocate",
    "to transform", "to convert", "to change", "to alter", "to modify", "to adjust",
    "to adapt", "to tailor", "to customize", "to personalize", "to individualize", "to specialize",
    "to focus", "to target", "to direct", "to orient", "to align", "to synchronize",
    "to harmonize", "to coordinate", "to integrate", "to incorporate", "to combine", "to merge",
    "to unite", "to join", "to connect", "to link", "to bridge", "to bond",
    "to attach", "to affix", "to secure", "to fasten", "to anchor", "to stabilize",
    "to balance", "to equalize", "to normalize", "to standardize", "to formalize", "to regularize",
    "to systematize", "to methodize", "to organize", "to structure", "to arrange", "to order",
    "to sequence", "to prioritize", "to rank", "to grade", "to classify", "to categorize",
    "to group", "to cluster", "to segment", "to partition", "to divide", "to separate",
    "to isolate", "to extract", "to remove", "to eliminate", "to eradicate", "to abolish",
    "to terminate", "to end", "to conclude", "to finish", "to complete", "to accomplish",
    "to achieve", "to attain", "to realize", "to fulfill", "to satisfy", "to meet",
    "to match", "to equal", "to surpass", "to exceed", "to outperform", "to outdo",
    "to outshine", "to outclass", "to outmatch", "to outstrip", "to outpace", "to outdistance",
    
    # Phrases with "resulting in"
    "resulting in", "resulting in better", "resulting in improved", "resulting in enhanced",
    "resulting in increased", "resulting in decreased", "resulting in reduced", "resulting in minimized",
    "resulting in maximized", "resulting in optimized", "resulting in streamlined", "resulting in simplified",
    "resulting in accelerated", "resulting in faster", "resulting in quicker", "resulting in speedier",
    "resulting in more efficient", "resulting in more effective", "resulting in more reliable",
    "resulting in more secure", "resulting in more robust", "resulting in more scalable",
    "resulting in more maintainable", "resulting in more performant", "resulting in more suitable",
    "resulting in more appropriate", "resulting in more applicable", "resulting in more advantageous",
    
    # Phrases with "leading to"
    "leading to", "leading to better", "leading to improved", "leading to enhanced",
    "leading to increased", "leading to decreased", "leading to reduced", "leading to minimized",
    "leading to maximized", "leading to optimized", "leading to streamlined", "leading to simplified",
    "leading to accelerated", "leading to faster", "leading to quicker", "leading to speedier",
    "leading to more efficient", "leading to more effective", "leading to more reliable",
    "leading to more secure", "leading to more robust", "leading to more scalable",
    "leading to more maintainable", "leading to more performant", "leading to more suitable",
    "leading to more appropriate", "leading to more applicable", "leading to more advantageous",
    
    # Phrases with "contributing to"
    "contributing to", "contributing to better", "contributing to improved", "contributing to enhanced",
    "contributing to increased", "contributing to decreased", "contributing to reduced", "contributing to minimized",
    "contributing to maximized", "contributing to optimized", "contributing to streamlined", "contributing to simplified",
    "contributing to accelerated", "contributing to faster", "contributing to quicker", "contributing to speedier",
    "contributing to more efficient", "contributing to more effective", "contributing to more reliable",
    "contributing to more secure", "contributing to more robust", "contributing to more scalable",
    "contributing to more maintainable", "contributing to more performant", "contributing to more suitable",
    "contributing to more appropriate", "contributing to more applicable", "contributing to more advantageous"
   ],
    
    # Technical recommendation bullet point markers
"bullet_point_markers": [
    # Basic bullet point markers (common in technical docs)
    "• ", "* ", "- ", "○ ", "◦ ", "· ", "» ", "› ", "‣ ", "⁃ ", "⦿ ", "⦾ ", "⁌ ", "⁍ ",
    
    # Simple geometric markers (common in technical docs)
    "□ ", "■ ", "◇ ", "◆ ", "△ ", "▲ ", "▽ ", "▼ ", "○ ", "● ", "◎ ", "◯ ",
    
    # Arrow markers (useful for technical steps)
    "→ ", "⇒ ", "⟹ ", "⟶ ", "⟾ ", "⟼ ", "⟿ ", "⤇ ", "⥤ ", "⥢ ", "⥱ ", "➔ ", "➙ ", "➛ ", "➜ ", "➝ ", "➞ ", "➟ ", "➠ ", "➡ ", "➢ ", "➣ ", "➤ ", "➥ ", "➦ ", "➧ ", "➨ ", "➩ ", "➪ ",
    
    # Numbered and lettered list markers (very common in technical docs)
    "1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. ", "10. ",
    "a. ", "b. ", "c. ", "d. ", "e. ", "f. ", "g. ", "h. ", "i. ", "j. ",
    "A. ", "B. ", "C. ", "D. ", "E. ", "F. ", "G. ", "H. ", "I. ", "J. ",
    "i) ", "ii) ", "iii) ", "iv) ", "v) ", "vi) ", "vii) ", "viii) ", "ix) ", "x) ",
    "I. ", "II. ", "III. ", "IV. ", "V. ", "VI. ", "VII. ", "VIII. ", "IX. ", "X. ",
    "(1) ", "(2) ", "(3) ", "(4) ", "(5) ", "(6) ", "(7) ", "(8) ", "(9) ", "(10) ",
    "(a) ", "(b) ", "(c) ", "(d) ", "(e) ", "(f) ", "(g) ", "(h) ", "(i) ", "(j) ",
    "(A) ", "(B) ", "(C) ", "(D) ", "(E) ", "(F) ", "(G) ", "(H) ", "(I) ", "(J) ",
    "1) ", "2) ", "3) ", "4) ", "5) ", "6) ", "7) ", "8) ", "9) ", "10) ",
    "a) ", "b) ", "c) ", "d) ", "e) ", "f) ", "g) ", "h) ", "i) ", "j) ",
    "A) ", "B) ", "C) ", "D) ", "E) ", "F) ", "G) ", "H) ", "I) ", "J) ",
    
    # Technical document specific markers
    "Step 1: ", "Step 2: ", "Step 3: ", "Step 4: ", "Step 5: ",
    "Note: ", "Important: ", "Warning: ", "Caution: ", "Tip: ", "Info: ", "Key Point: ",
    "Recommendation: ", "Best Practice: ", "Guideline: ", "Rule: ", "Principle: ",
    "Requirement: ", "Specification: ", "Constraint: ", "Limitation: ", "Dependency: ",
    "Prerequisite: ", "Assumption: ", "Condition: ", "Parameter: ", "Configuration: ",
    "Option: ", "Setting: ", "Property: ", "Attribute: ", "Feature: ", "Function: ",
    "Method: ", "Procedure: ", "Process: ", "Workflow: ", "Pipeline: ", "Sequence: ",
    "Algorithm: ", "Pattern: ", "Architecture: ", "Design: ", "Implementation: ",
    "Example: ", "Sample: ", "Case: ", "Scenario: ", "Use Case: ", "Test Case: ",
    
    # Technical prefixed numbers (common in specifications)
    "R1: ", "R2: ", "R3: ", "R4: ", "R5: ", # Requirements
    "T1: ", "T2: ", "T3: ", "T4: ", "T5: ", # Tasks
    "F1: ", "F2: ", "F3: ", "F4: ", "F5: ", # Features
    "C1: ", "C2: ", "C3: ", "C4: ", "C5: ", # Constraints
    "S1: ", "S2: ", "S3: ", "S4: ", "S5: ", # Steps
    "P1: ", "P2: ", "P3: ", "P4: ", "P5: ", # Principles
    "BP1: ", "BP2: ", "BP3: ", "BP4: ", "BP5: ", # Best Practices
    "REC1: ", "REC2: ", "REC3: ", "REC4: ", "REC5: ", # Recommendations
    
    # Technical document section markers
    "Section 1: ", "Section 2: ", "Section 3: ", "Section 4: ", "Section 5: ",
    "Chapter 1: ", "Chapter 2: ", "Chapter 3: ", "Chapter 4: ", "Chapter 5: ",
    "Part 1: ", "Part 2: ", "Part 3: ", "Part 4: ", "Part 5: ",
    "Module 1: ", "Module 2: ", "Module 3: ", "Module 4: ", "Module 5: ",
    "Component 1: ", "Component 2: ", "Component 3: ", "Component 4: ", "Component 5: ",
    "Layer 1: ", "Layer 2: ", "Layer 3: ", "Layer 4: ", "Layer 5: ",
    "Tier 1: ", "Tier 2: ", "Tier 3: ", "Tier 4: ", "Tier 5: ",
    "Level 1: ", "Level 2: ", "Level 3: ", "Level 4: ", "Level 5: ",
    
    # Technical checklist markers
    "[ ] ", "[x] ", "[X] ", "[✓] ", "[✔] ", "[✗] ", "[✘] ", "[o] ", "[O] ", "[•] "
]


}

# Flatten all recommendation indicators into a single list for easier checking
ALL_RECOMMENDATION_INDICATORS = []
for category in RECOMMENDATION_INDICATORS.values():
    ALL_RECOMMENDATION_INDICATORS.extend(category)

# Lines to ignore during processing
IGNORED_LINES = [
    "Maintainer", 
    "Publication date", 
    "tags:",
    "Contribute on GitLab", 
    "Discuss on plazza", 
    "See contacts"
]

# Section-specific keywords for classification
SECTION_KEYWORDS = {
    "Architecture": [
        "architecture", "structure", "design", "component", "system", "pattern", "microservice",
        "monolith", "service", "tier", "layer", "module", "framework", "infrastructure",
        "platform", "solution", "topology", "organization", "composition", "arrangement",
        "configuration", "setup", "layout", "blueprint", "model", "paradigm", "approach",
        "methodology", "strategy", "tactic", "technique", "mechanism", "scheme", "plan"
    ],
    "Communications": [
        "communication", "network", "protocol", "api", "rest", "grpc", "message", "event",
        "notification", "alert", "signal", "broadcast", "publish", "subscribe", "queue",
        "topic", "channel", "stream", "flow", "transmission", "exchange", "interaction",
        "interface", "endpoint", "connection", "link", "bridge", "gateway", "proxy",
        "router", "switch", "hub", "node", "client", "server", "peer", "socket", "port"
    ],
    "Cache": [
        "cache", "caching", "redis", "memcached", "in-memory", "performance", "speed",
        "latency", "throughput", "response time", "load time", "access time", "hit rate",
        "miss rate", "eviction", "expiration", "ttl", "time to live", "invalidation",
        "refresh", "update", "synchronization", "consistency", "coherence", "distributed",
        "local", "global", "shared", "private", "public", "client-side", "server-side",
        "browser", "cdn", "content delivery", "edge", "proxy", "reverse proxy"
    ],
    "Programming languages and execution environment": [
        "language", "java", "python", "javascript", "typescript", "go", "rust", "c#", "coding",
        "programming", "development", "software", "application", "app", "code", "codebase",
        "repository", "version", "release", "build", "compile", "interpret", "execute",
        "runtime", "environment", "platform", "framework", "library", "package", "module",
        "dependency", "import", "include", "require", "use", "utilize", "implement", "extend",
        "inherit", "override", "interface", "abstract", "concrete", "class", "object", "function",
        "method", "procedure", "routine", "algorithm", "logic", "control flow", "data flow"
    ],
    "Data": [
        "data", "database", "sql", "nosql", "storage", "persistence", "mongodb", "postgresql",
        "mysql", "oracle", "sqlserver", "cassandra", "dynamodb", "cosmosdb", "elasticsearch",
        "solr", "neo4j", "graph", "relational", "document", "key-value", "column", "time-series",
        "search", "index", "query", "transaction", "acid", "base", "consistency", "availability",
        "partition tolerance", "sharding", "replication", "backup", "restore", "recovery",
        "migration", "schema", "model", "entity", "attribute", "field", "column", "row", "record",
        "table", "collection", "document", "store", "warehouse", "lake", "mart", "etl", "elt"
    ],
    "Development, test and integration tools": [
        "development", "test", "testing", "integration", "ci/cd", "pipeline", "tool", "ide",
        "editor", "environment", "framework", "library", "package", "module", "dependency",
        "build", "compile", "lint", "format", "style", "convention", "standard", "guideline",
        "practice", "methodology", "process", "procedure", "workflow", "lifecycle", "agile",
        "scrum", "kanban", "waterfall", "iterative", "incremental", "continuous", "automation",
        "script", "configuration", "setup", "installation", "deployment", "release", "version",
        "control", "git", "svn", "mercurial", "branch", "merge", "pull", "push", "commit",
        "review", "approve", "reject", "feedback", "comment", "discussion", "collaboration"
    ],
    "Operations and End of Life": [
        "operation", "deployment", "monitor", "logging", "kubernetes", "docker", "container",
        "orchestration", "automation", "configuration", "management", "administration",
        "maintenance", "support", "service", "availability", "reliability", "stability",
        "resilience", "robustness", "durability", "performance", "efficiency", "optimization",
        "scaling", "load balancing", "failover", "backup", "restore", "recovery", "disaster",
        "incident", "problem", "change", "release", "version", "update", "upgrade", "patch",
        "fix", "hotfix", "security", "vulnerability", "threat", "risk", "compliance", "audit",
        "log", "metric", "alert", "notification", "dashboard", "report", "analysis", "insight",
        "end of life", "eol", "sunset", "decommission", "retire", "archive", "delete", "purge"
    ]
}

def enhanced_rule_based_classification(text: str) -> Tuple[bool, float]:
    """
    Use enhanced rules to classify if text contains a recommendation.
    
    Args:
        text: The text to classify
        
    Returns:
        Tuple of (is_recommendation, confidence_score)
    """
    if not text.strip():
        return False, 0.0
        
    text_lower = text.lower()
    
    # Check for explicit recommendation phrases (highest confidence)
    for phrase in RECOMMENDATION_INDICATORS["explicit_recommendations"]:
        if phrase in text_lower:
            return True, 0.95
    
    # Check for obligation phrases (high confidence)
    for phrase in RECOMMENDATION_INDICATORS["obligation_phrases"]:
        if phrase in text_lower:
            return True, 0.9
    
    # Check for best practice phrases (high confidence)
    for phrase in RECOMMENDATION_INDICATORS["best_practice_phrases"]:
        if phrase in text_lower:
            return True, 0.85
    
    # Check for preference phrases (medium-high confidence)
    for phrase in RECOMMENDATION_INDICATORS["preference_phrases"]:
        if phrase in text_lower:
            return True, 0.8
    
    # Check for imperative verbs at the beginning of the text (medium confidence)
    for verb in RECOMMENDATION_INDICATORS["imperative_verbs"]:
        if text_lower.startswith(verb):
            return True, 0.75
    
    # Check for bullet points with imperative verbs (medium confidence)
    for bullet in RECOMMENDATION_INDICATORS["bullet_point_markers"]:
        if text.startswith(bullet):
            words = text_lower[len(bullet):].strip().split()
            if words and any(words[0].startswith(verb.strip()) for verb in RECOMMENDATION_INDICATORS["imperative_verbs"]):
                return True, 0.7
    
    # Check for consequence phrases (lower confidence)
    for phrase in RECOMMENDATION_INDICATORS["consequence_phrases"]:
        if phrase in text_lower:
            return True, 0.6
    
    # Check for sentences with recommendation-like structure
    if re.search(r'(it is|is|are) (important|recommended|suggested|advised|preferable|better|best) to', text_lower):
        return True, 0.8
    
    # Check for negative patterns (avoid, don't, etc.)
    if re.search(r'(avoid|don\'t|do not|never) (use|implement|deploy|choose|select)', text_lower):
        return True, 0.85
    
    # Check for positive patterns (always use, etc.)
    if re.search(r'(always|consistently) (use|implement|deploy|choose|select)', text_lower):
        return True, 0.85
    
    # Check for comparative recommendations
    if re.search(r'(use|choose|select|prefer) [a-z\s]+ (over|instead of|rather than) [a-z\s]+', text_lower):
        return True, 0.8
    
    # Check for numbered lists that might be recommendations
    if re.match(r'^\d+\.\s', text) and any(verb.strip() in text_lower.split()[1:3] for verb in RECOMMENDATION_INDICATORS["imperative_verbs"]):
        return True, 0.65
    
    # Lower confidence checks for technical recommendations
    if any(tech_term in text_lower for tech_term in ["configuration", "setting", "parameter", "option", "flag", "property"]):
        if any(verb.strip() in text_lower for verb in RECOMMENDATION_INDICATORS["imperative_verbs"]):
            return True, 0.6
    
    return False, 0.1

def determine_best_section(text: str, current_section: str) -> str:
    """
    Use rule-based approach to determine which section a text belongs to.
    
    Args:
        text: The text to classify
        current_section: The current section being processed
        
    Returns:
        The section name the text belongs to
    """
    if not text.strip():
        return current_section
        
    text_lower = text.lower()
    
    # Check for section-specific keywords
    best_match = current_section
    best_score = 0
    
    for section_name, keywords in SECTION_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Weight exact matches higher
        for keyword in keywords:
            if f" {keyword} " in f" {text_lower} ":  # Add spaces to ensure whole word match
                score += 1
                
        # If we found a better match, update
        if score > best_score:
            best_score = score
            best_match = section_name
            
            # Find the full section name that matches this section_name
            for full_section in SECTIONS:
                if section_name.lower() in full_section.lower():
                    best_match = full_section
                    break
    
    # If we have a good match (at least 2 keywords), use it
    if best_score >= 2:
        return best_match
    
    # Otherwise, stick with the current section
    return current_section

def clean_docx(docx_path: str) -> Dict[str, List[str]]:
    """
    Reads a Word document and extracts structured recommendations with enhanced rule-based classification.
    
    Args:
        docx_path: Path to the Word document
        
    Returns:
        A dictionary with sections as keys and lists of recommendations as values
    
    Raises:
        FileNotFoundError: If the specified document doesn't exist
        Exception: For other document processing errors
    """
    try:
        logger.info(f"Loading document: {docx_path}")
        doc = Document(docx_path)
    except FileNotFoundError:
        logger.error(f"Error: File {docx_path} not found")
        raise
    except Exception as e:
        logger.error(f"Error opening document: {e}")
        raise

    # Initialize tracking variables
    current_section = None  # Track which section we're currently in
    in_recommendation = False  # Flag to track if we're processing a recommendation block
    sections_data = {sec: [] for sec in SECTIONS}  # Initialize result dictionary
    
    # Process each paragraph in the document
    try:
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue  # Skip empty paragraphs

            # Skip lines that match our ignore patterns
            if any(ignored.lower() in text.lower() for ignored in IGNORED_LINES):
                continue

            # Check if this paragraph is a section header (case insensitive)
            if text.lower() in [section.lower() for section in SECTIONS]:
                current_section = next(section for section in SECTIONS 
                                      if section.lower() == text.lower())
                logger.debug(f"Found section: {current_section}")
                in_recommendation = False
                continue

            # Skip if we haven't found a valid section yet
            if not current_section:
                continue

            # Check for explicit recommendation markers
            if text.lower() in ["recommendation", "recommendation:"]:
                in_recommendation = True
                continue

            # If we're in a recommendation block, add the text and reset the flag
            if in_recommendation:
                # Use enhanced rule-based classification
                is_rec, confidence = enhanced_rule_based_classification(text)
                if is_rec or confidence > 0.5:
                    sections_data[current_section].append(text)
                    logger.debug(f"Added recommendation to {current_section}: {text[:50]}...")
                in_recommendation = False
                continue

            # Check for inline recommendations by splitting into sentences
            clean_sentences = re.split(r'(?<=[.!?]) +', text)
            for sent in clean_sentences:
                sent = sent.strip()
                if not sent:
                    continue
                    
                # Use enhanced rule-based classification
                is_rec, confidence = enhanced_rule_based_classification(sent)
                
                # Add if identified as recommendation
                if is_rec or confidence > 0.5:
                    # Determine the best section
                    best_section = determine_best_section(sent, current_section)
                    sections_data[best_section].append(sent)
                    logger.debug(f"Added recommendation to {best_section}: {sent[:50]}...")

        logger.info("Document cleaning and extraction completed successfully")
        
        # Count extracted recommendations
        total_recommendations = sum(len(recs) for recs in sections_data.values())
        logger.info(f"Extracted {total_recommendations} recommendations across {len(SECTIONS)} sections")
        
        return sections_data
        
    except Exception as e:
        logger.error(f"Error during document processing: {e}")
        raise

def save_to_json(data: Dict[str, List[str]], output_path: str) -> None:
    """
    Saves the extracted recommendations to a JSON file.
    
    Args:
        data: Dictionary containing the extracted recommendations
        output_path: Path where the JSON file will be saved
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results to {output_path}: {e}")
        raise

def main():
    """Main function to parse arguments and run the extraction process."""
    parser = argparse.ArgumentParser(
        description='Extract recommendations from Word documents for RAG processing with enhanced rule-based classification'
    )
    parser.add_argument(
        '--input', '-i', 
        default="recommendation.docx", 
        help='Path to Word document'
    )
    parser.add_argument(
        '--output', '-o', 
        help='Output JSON file path'
    )
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Determine output path if not specified
    output_path = args.output
    if not output_path:
        input_path = Path(args.input)
        output_path = input_path.with_suffix('.json')
    
    try:
        # Process the document
        data = clean_docx(args.input)
        
        # Save or print results
        if output_path:
            save_to_json(data, output_path)
        else:
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
        return 0
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
