                 
ONSETS = ["", "", "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v", "w", "y", "z",
          "bl", "br", "ch", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "sh", "sl", "sm", "sn", "sp", "st",
          "str", "th", "tr", "wh"]
NUCLEI = ["a", "e", "i", "o", "u", "a", "e", "i", "o", "ai", "ea", "ee", "ie", "oo", "ou", "oa", "y"]
CODAS = ["", "", "", "n", "r", "s", "t", "l", "m", "d", "nd", "nt", "st", "ck", "ng", "rd", "ll", "ss", "x", "rk"]
COMMON_ZH = (
    "的一是在不了有和人這中大為上個國我以要他時來用們生到作地於出就分對成會可主發年動同工也能下過子說產種面"
    "而方後多定行學法所民得經十三之進著等部度家電力裡如水化高自二理起小物現實加量都兩體制機當使點從業本去把性"
    "好應開它合還因由其些然前外天政四日那社義事平形相全表間樣與關各重新線內數正心反你明看原又麼利比或但質氣第"
    "向道命此變條只沒結解問意建月公無系軍很情者最立代想已通並提直題黨程展五果料象員革位入常文總次品式活設及管"
    "特件長求老頭基資邊流路級少圖山統接知較將組見計別她手角期根論運農指幾九區強放決西被幹做必戰先回則任取據處"
    "理世車收費單據合約報告申請表格姓名地址電話日期金額備註簽章統一編號訂單發票收據證明書")


class TextGen:
    def __init__(self, rng):
        self.r = rng


    def word(self, min_syl=1, max_syl=3):
        r = self.r
        n = r.choices(range(min_syl, max_syl + 1), weights=[4, 3, 2, 1][: max_syl - min_syl + 1])[0]
        return "".join(r.choice(ONSETS) + r.choice(NUCLEI) + (r.choice(CODAS) if k == n - 1 or r.random() < 0.3 else "")
                       for k in range(n))

    def words(self, n, caps=False):
        ws = [self.word() for _ in range(n)]
        return " ".join(w.capitalize() for w in ws) if caps else " ".join(ws)

    def sentence(self, n=None):
        r = self.r
        n = n or r.randint(4, 14)
        ws = [self.word() for _ in range(n)]
        for k in range(1, n - 1):
            if r.random() < 0.08:
                ws[k] += ","
        s = " ".join(ws)
        return s[0].upper() + s[1:] + r.choice([".", ".", ".", "?", "!", ":"])

    def line(self, max_chars):
        out = ""
        while True:
            s = self.sentence()
            if len(out) + len(s) + 1 > max_chars:
                return (out or s[:max_chars]).strip()
            out += s + " "

    def name(self):
        return f"{self.word(1, 2).capitalize()} {self.word(1, 3).capitalize()}"

    def company(self):
        r = self.r
        core = self.words(r.randint(1, 2), caps=True)
        suffix = self.word(1, 1).capitalize() if r.random() < 0.5 else ""
        tail = r.choice(["", "", " & " + self.word(1, 2).capitalize(), " " + "".join(r.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(r.randint(2, 3)))])
        return (core + (" " + suffix if suffix else "") + tail).strip()

    def headline(self):
        r = self.r
        k = r.random()
        if k < 0.25:
            return f"{r.randint(1, 9) * 10}% {self.word(1, 1).upper()}"
        if k < 0.4:
            return f"{self.word(1, 2).upper()} {self.money()}"
        return self.words(r.randint(1, 3)).upper() if r.random() < 0.7 else self.words(r.randint(1, 3), caps=True)

    def phone(self):
        r = self.r
        d = lambda n: "".join(str(r.randint(0, 9)) for _ in range(n))
        return r.choice([f"+{r.randint(1, 99)} {d(3)} {d(3)} {d(4)}", f"({d(3)}) {d(3)}-{d(4)}", f"{d(2)}-{d(4)}-{d(4)}"])

    def email(self):
        return f"{self.word(1, 2)}{self.r.choice(['.', '_', ''])}{self.word(1, 2)}@{self.word(1, 2)}.{self.r.choice(['com', 'net', 'org', 'io', 'co'])}"

    def web(self):
        return f"www.{self.word(1, 3)}.{self.r.choice(['com', 'net', 'shop', 'co'])}"

    def address(self):
        r = self.r
        return [f"{r.randint(1, 999)} {self.word(1, 2).capitalize()} {r.choice(['St', 'Ave', 'Rd', 'Blvd', 'Ln'])}",
                f"{self.word(1, 3).capitalize()}, {''.join(r.choice('ABCDEFGHKLMNPRSTUVWXY') for _ in range(2))} {r.randint(10000, 99999)}"]

    def date(self):
        r = self.r
        y, m, d = r.randint(1995, 2030), r.randint(1, 12), r.randint(1, 28)
        return r.choice([f"{y}-{m:02d}-{d:02d}", f"{d:02d}/{m:02d}/{y}", f"{m:02d}/{d:02d}/{y % 100:02d}"])

    def time(self):
        return f"{self.r.randint(0, 23):02d}:{self.r.randint(0, 59):02d}"

    def money(self, lo=1, hi=999):
        r = self.r
        v = r.uniform(lo, hi)
        return r.choice([f"${v:,.2f}", f"{v:,.2f}", f"${int(v)}", f"{r.choice(['EUR', 'USD', 'NT$', '£', '€'])}{v:,.0f}"])

    def code(self, n=None):
        r = self.r
        return "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(n or r.randint(5, 12)))


    def zh_word(self):
        return "".join(self.r.choice(COMMON_ZH) for _ in range(self.r.randint(2, 6)))

    def zh_line(self, max_chars):
        r = self.r
        out = ""
        while len(out) < max_chars - 1:
            out += self.zh_word() + (r.choice("，。、") if r.random() < 0.25 else "")
        return out[:max_chars]
