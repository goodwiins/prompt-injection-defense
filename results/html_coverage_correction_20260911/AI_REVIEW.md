# AI review of HTML correction candidates

Reviewed at 2026-09-11T22:17:49.471606+00:00. User-authorized AI review: **15 injection attempts, 30 provisional non-injections, 0 unresolved labels**. No model scores were consulted. These judgments agree with the proposed labels.

The same assistant prepared and reviewed this packet. This is not blind independent review, human sign-off, or evidence that the reviewer is a better calibrated classifier. The original human CSV remains unchanged.

Review covered all 15 unique parent reading aids, non-redundant text-bearing attributes, raw instruction-cue search, and all 30 exact inserted spans. No script/style/comment blocks were found by the parser. The only original-page cue hit was CarMax describing its site-chat button. Removing each insertion exactly restores its original parent; all insertion spans have a complete token window. This review is not exhaustive semantic inspection of every raw markup byte.

The 15 direct instructions change the browsing task to an essay, unrelated factual/tutorial response, anagramming, space removal or typos. The 15 matched quoted variants explicitly frame the same words as examples. Quoted framing remains a potential shortcut.

[Labels](ai_review.csv) · [Structural checks](ai_structural_review.json) · [Authorization and limitations](ai_authorization.json)

| Website | Original-page review |
|---|---|
| aa | Flight booking/status, calendar accessibility instructions, loyalty offers and travel links. |
| budget | Rental forms, pickup times, age/residency options, discounts and ordinary terms. |
| carmax | Vehicle shopping, financing estimates, returns, inventory and store links. The virtual-assistant attribute describes opening the site chat; it does not redirect an agent. |
| cvs | Pharmacy shopping, product categories, healthcare promotions and customer links. |
| delta | Flight search, trip/calendar controls, travel promotions and policies. |
| gamestop | Product categories, game and gift-card shopping, rewards and ordinary account metadata. |
| ign | Gaming/news headlines, reviews, entertainment categories, platform filters and advertising. |
| koa | Campground search, trip planning, camping advice, rewards, regional choices and reservations. |
| nfl | Football news, draft coverage, team links, fan polls, promotions and cookie settings. |
| thetrainline | Train and coach search, routes, ticket explanations and ordinary sign-in dialog. |
| ticketcenter | Event/team catalogs, ticket purchase explanations, event dates, testimonials and guarantees. |
| ultimate-guitar | Song/tab catalog, music headlines, guitar forum excerpts, user profiles and advertising. |
| uniqlo | Clothing promotions, product prices, collections, app features and store/customer links. |
| united | Air travel booking, calendar accessibility instructions, passenger controls, promotions and policies. |
| us.megabus | Bus route/date/traveler controls, trip help, employment and fare promotions. |

Proceed with the frozen balanced replacement: 256 training pages (128/128), 207 validation pages (101 attacks/106 negatives), six fixed epochs, seeds 17/42/73 and matched frozen controls. No final test is included. Historical human-review preflight remains a record of that separate stricter scope.
