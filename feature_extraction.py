import re
import math
from collections import Counter
from urllib.parse import urlparse, parse_qs


# ============================================================
# WORD LISTS
# ============================================================

URGENCY_WORDS = {
    "urgent", "immediately", "warning", "alert", "verify",
    "verification", "suspended", "blocked", "expire", "expired",
    "account", "login", "confirm", "security", "update"
}

SECURITY_WORDS = {
    "security", "secure", "password", "credential",
    "authentication", "verify", "verification", "account",
    "login", "signin", "bank", "payment"
}

BRAND_WORDS = {
    "paypal", "microsoft", "google", "facebook", "instagram",
    "apple", "amazon", "netflix", "linkedin", "twitter",
    "whatsapp", "bank"
}

SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq",
    ".top", ".xyz", ".click", ".link",
    ".work", ".zip"
}

SHORTENING_SERVICES = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co",
    "ow.ly", "is.gd", "buff.ly", "adf.ly",
    "bit.do", "cutt.ly", "shorturl.at",
    "rebrand.ly", "tiny.cc"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def entropy(text):
    """Calculate Shannon entropy."""
    if not text:
        return 0.0

    counts = Counter(text)
    length = len(text)

    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def count_words(text, word_set):
    """Count occurrences of words from a predefined set."""
    text = text.lower()

    return sum(
        len(re.findall(r"\b" + re.escape(word) + r"\b", text))
        for word in word_set
    )


def is_ip_address(hostname):
    """Check whether hostname is an IPv4 address."""

    if not hostname:
        return 0

    pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"

    if not re.match(pattern, hostname):
        return 0

    parts = hostname.split(".")

    try:
        return int(all(0 <= int(part) <= 255 for part in parts))
    except ValueError:
        return 0


def get_domain_parts(hostname):
    """Return hostname components."""

    if not hostname:
        return []

    return [part for part in hostname.lower().split(".") if part]


def get_subdomains(hostname):
    """
    Approximate subdomains by removing the final domain and TLD.
    Example:
        login.accounts.google.com
        -> ['login', 'accounts']
    """

    parts = get_domain_parts(hostname)

    if len(parts) <= 2:
        return []

    return parts[:-2]


def has_keyword(text, keyword):
    return keyword in text.lower()


# ============================================================
# MAIN FEATURE EXTRACTION
# ============================================================

def extract_features(url):
    """
    Extract the 63 numeric features used by the binary model.

    Target mapping used during training:

        0 = Safe / benign
        1 = Malicious
    """

    url = str(url).strip()

    if not url:
        raise ValueError("URL cannot be empty.")

    # --------------------------------------------------------
    # URL normalization
    # --------------------------------------------------------

    parse_url = url

    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        parse_url
    ):
        parse_url = "http://" + parse_url

    parsed = urlparse(parse_url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    hostname_lower = hostname.lower()
    url_lower = url.lower()

    features = {}

    # ========================================================
    # WEB FEATURES
    # ========================================================

    # These features require inspecting the actual webpage.
    # The current model was trained using them, but we are
    # keeping safe defaults here until live webpage inspection
    # is added.

    features["web_is_live"] = 0
    features["web_security_score"] = 0
    features["web_forms_count"] = 0
    features["web_password_fields"] = 0
    features["web_has_login"] = 0
    features["web_ssl_valid"] = 1 if parsed.scheme.lower() == "https" else 0

    # ========================================================
    # BASIC URL FEATURES
    # ========================================================

    features["url_len"] = len(url)

    features["@"] = url.count("@")
    features["?"] = url.count("?")
    features["-"] = url.count("-")
    features["="] = url.count("=")
    features["."] = url.count(".")
    features["#"] = url.count("#")
    features["%"] = url.count("%")
    features["+"] = url.count("+")
    features["$"] = url.count("$")
    features["!"] = url.count("!")
    features["*"] = url.count("*")
    features[","] = url.count(",")
    features["//"] = url.count("//")

    features["digits"] = sum(c.isdigit() for c in url)
    features["letters"] = sum(c.isalpha() for c in url)

    # ========================================================
    # BASIC SECURITY FEATURES
    # ========================================================

    features["abnormal_url"] = int(
        not url_lower.startswith(("http://", "https://"))
    )

    features["https"] = int(
        parsed.scheme.lower() == "https"
    )

    features["Shortining_Service"] = int(
        hostname_lower in SHORTENING_SERVICES
        or any(
            hostname_lower.endswith("." + domain)
            for domain in SHORTENING_SERVICES
        )
    )

    features["having_ip_address"] = is_ip_address(hostname)

    # ========================================================
    # DEFACEMENT FEATURES
    # ========================================================

    hacked_terms = {
        "hacked",
        "hack",
        "defaced",
        "deface",
        "owned",
        "pwned",
        "cyber",
        "attacked",
        "attack"
    }

    suspicious_extensions = {
        ".php",
        ".asp",
        ".aspx",
        ".jsp",
        ".cgi",
        ".exe",
        ".sh",
        ".bat"
    }

    path_lower = path.lower()
    full_lower = url_lower

    features["defac_has_hacked_terms"] = int(
        any(term in full_lower for term in hacked_terms)
    )

    features["defac_has_suspicious_ext"] = int(
        any(path_lower.endswith(ext) for ext in suspicious_extensions)
    )

    path_parts = [
        part for part in path.split("/")
        if part
    ]

    features["defac_path_depth"] = len(path_parts)

    features["defac_is_deep_path"] = int(
        len(path_parts) >= 5
    )

    features["defac_path_underscores"] = path.count("_")

    features["defac_is_gov_edu"] = int(
        hostname_lower.endswith(".gov")
        or hostname_lower.endswith(".edu")
        or ".gov." in hostname_lower
        or ".edu." in hostname_lower
    )

    features["defac_has_index_php"] = int(
        "index.php" in path_lower
    )

    query_params = parse_qs(query)

    features["defac_has_option_param"] = int(
        "option" in query_params
        or "option=" in query_lower_safe(query)
    )

    # ========================================================
    # PHISHING FEATURES
    # ========================================================

    features["phish_has_brand"] = int(
        any(
            re.search(
                r"\b" + re.escape(brand) + r"\b",
                full_lower
            )
            for brand in BRAND_WORDS
        )
    )

    subdomains = get_subdomains(hostname_lower)

    features["phish_brand_in_subdomain"] = int(
        any(
            brand in subdomain
            for brand in BRAND_WORDS
            for subdomain in subdomains
        )
    )

    features["phish_brand_in_path"] = int(
        any(
            brand in path_lower
            for brand in BRAND_WORDS
        )
    )

    features["phish_hyphen_count"] = hostname.count("-")

    features["phish_digit_count"] = sum(
        c.isdigit() for c in hostname
    )

    features["phish_long_domain"] = int(
        len(hostname) > 30
    )

    features["phish_many_subdomains"] = int(
        len(subdomains) >= 2
    )

    features["phish_suspicious_tld"] = int(
        any(
            hostname_lower.endswith(tld)
            for tld in SUSPICIOUS_TLDS
        )
    )

    features["phish_keyword_count"] = (
        count_words(url, URGENCY_WORDS)
        + count_words(url, SECURITY_WORDS)
    )

    features["phish_has_redirect"] = int(
        "redirect" in full_lower
        or "url=" in full_lower
        or "redirect=" in full_lower
    )

    features["phish_param_count"] = (
        len(parse_qs(query))
        if query
        else 0
    )

    features["phish_encoded_chars"] = len(
        re.findall(r"%[0-9a-fA-F]{2}", url)
    )

    # ========================================================
    # ENHANCED PHISHING FEATURES
    # ========================================================

    features["enh_urgency_count"] = count_words(
        url,
        URGENCY_WORDS
    )

    features["enh_security_count"] = count_words(
        url,
        SECURITY_WORDS
    )

    features["enh_brand_count"] = count_words(
        url,
        BRAND_WORDS
    )

    # Brand hijacking approximation
    brand_hijack = 0

    for brand in BRAND_WORDS:
        if brand in hostname_lower:
            legitimate_domains = {
                f"{brand}.com",
                f"www.{brand}.com"
            }

            if hostname_lower not in legitimate_domains:
                brand_hijack = 1
                break

    features["enh_brand_hijack"] = brand_hijack

    features["enh_subdomain_count"] = len(subdomains)

    features["enh_long_path"] = int(
        len(path) > 75
    )

    parameter_count = (
        len(parse_qs(query))
        if query
        else 0
    )

    features["enh_many_params"] = int(
        parameter_count >= 5
    )

    features["enh_suspicious_tld"] = int(
        any(
            hostname_lower.endswith(tld)
            for tld in SUSPICIOUS_TLDS
        )
    )

    # ========================================================
    # ADVANCED FEATURES
    # ========================================================

    features["adv_domain_ngram_entropy"] = entropy(
        hostname_lower
    )

    features["adv_path_entropy"] = entropy(
        path
    )

    alpha_chars = [
        c.lower()
        for c in url
        if c.isalpha()
    ]

    if alpha_chars:

        vowels = sum(
            c in "aeiou"
            for c in alpha_chars
        )

        consonants = sum(
            c not in "aeiou"
            for c in alpha_chars
        )

        features["adv_consonant_ratio"] = (
            consonants / len(alpha_chars)
        )

        features["adv_vowel_ratio"] = (
            vowels / len(alpha_chars)
        )

    else:

        features["adv_consonant_ratio"] = 0.0
        features["adv_vowel_ratio"] = 0.0

    features["adv_digit_ratio"] = (
        features["digits"] / len(url)
        if url
        else 0.0
    )

    features["adv_subdomain_count"] = len(
        subdomains
    )

    if subdomains:

        features["adv_avg_subdomain_len"] = (
            sum(
                len(x)
                for x in subdomains
            )
            / len(subdomains)
        )

    else:

        features["adv_avg_subdomain_len"] = 0.0

    tokens = [
        token
        for token in re.split(
            r"[^A-Za-z0-9]+",
            url
        )
        if token
    ]

    features["adv_token_count"] = len(tokens)

    if tokens:

        features["adv_avg_token_length"] = (
            sum(
                len(token)
                for token in tokens
            )
            / len(tokens)
        )

    else:

        features["adv_avg_token_length"] = 0.0

    return features


def query_lower_safe(query):
    """Helper used for checking query strings."""
    return query.lower()