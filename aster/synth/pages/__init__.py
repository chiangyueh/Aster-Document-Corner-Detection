                  
from . import business_card, document, flyer, postcard, receipt, sign, ticket


LAYOUTS = {
    "letter": (0.22, [(1.30, 1.45)], document.render_text, True),
    "landscape": (0.05, [(0.68, 0.78)], document.render_text, True),
    "book": (0.07, [(1.45, 1.60)], document.render_text, True),
    "note": (0.07, [(0.9, 1.45)], document.render_note, False),
    "blank": (0.05, [(0.7, 1.45)], document.render_blank, True),
    "receipt": (0.12, [(1.8, 4.0)], receipt.render, False),
    "business_card": (0.09, [(0.55, 0.64), (1.55, 1.8)], business_card.render, False),
    "postcard": (0.08, [(0.64, 0.72), (1.4, 1.55)], postcard.render, False),
    "ticket": (0.05, [(0.33, 0.5)], ticket.render, False),
    "sign": (0.06, [(0.2, 0.6), (1.4, 2.5)], sign.render, False),
    "flyer": (0.14, [(1.3, 1.5), (0.68, 0.75), (2.0, 2.2), (0.95, 1.05)], flyer.render, False),
}
