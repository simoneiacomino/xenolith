#include "cpu_policy.h"

#include <ctype.h>
#include <string.h>

/* A catalog match is deliberately exact. CPU brand names are identity hints,
 * not evidence that the machine has the memory or power conditions measured. */
typedef struct {
    const char *sku, *brand_prefix;
    int homogeneous, p, atom, use_homogeneous_atom, use_homogeneous_p;
    int take_homogeneous, take_p, take_atom;
    int shared_l3_atom_count; /* Zero means no L3-based Atom selection. */
    xe_cpu_policy_evidence evidence;
} policy;

static const policy policies[] = {
    {.sku="i7-1185g7", .brand_prefix="core",
     .homogeneous=4, .take_homogeneous=4,
     .evidence=XE_CPU_EVIDENCE_SPEC},
    {.sku="i7-11700", .brand_prefix="core",
     .homogeneous=8, .take_homogeneous=6,
     .evidence=XE_CPU_EVIDENCE_SPEC},
    {.sku="i7-1260p", .brand_prefix="core",
     .p=4, .atom=8, .take_p=4,
     .evidence=XE_CPU_EVIDENCE_SPEC},
    {.sku="i3-1220p", .brand_prefix="core",
     .p=2, .atom=8, .take_p=2, .take_atom=4,
     .evidence=XE_CPU_EVIDENCE_SPEC},
    {.sku="i5-12400", .brand_prefix="core",
     .p=6, .use_homogeneous_p=1, .take_p=6,
     .evidence=XE_CPU_EVIDENCE_SPEC},
    {.sku="i5-12600k", .brand_prefix="core",
     .p=6, .atom=4, .take_p=6,
     .evidence=XE_CPU_EVIDENCE_SPEC},
    {.sku="n100", .brand_prefix="n",
     .atom=4, .use_homogeneous_atom=1, .take_atom=4,
     .evidence=XE_CPU_EVIDENCE_SPEC},
    {.sku="n150", .brand_prefix="n",
     .atom=4, .use_homogeneous_atom=1, .take_atom=4,
     .evidence=XE_CPU_EVIDENCE_SPEC},
    {.sku="i7-1355u", .brand_prefix="core",
     .p=2, .atom=8, .take_p=2, .take_atom=4,
     .evidence=XE_CPU_EVIDENCE_SPEC},
    {.sku="i7-13700h", .brand_prefix="core",
     .p=6, .atom=8, .take_p=6,
     .evidence=XE_CPU_EVIDENCE_MEASURED_PROFILE},
    {.sku="i7-13700", .brand_prefix="core",
     .p=8, .atom=8, .take_p=6,
     .evidence=XE_CPU_EVIDENCE_SPEC},
    {.sku="155h", .brand_prefix="ultra7",
     .p=6, .atom=10, .take_p=6,
     .evidence=XE_CPU_EVIDENCE_SPEC},
    {.sku="125u", .brand_prefix="ultra5",
     .p=2, .atom=10, .take_p=2, .take_atom=4, .shared_l3_atom_count=8,
     .evidence=XE_CPU_EVIDENCE_SPEC},
    {.sku="255h", .brand_prefix="ultra7",
     .p=6, .atom=10, .take_p=6,
     .evidence=XE_CPU_EVIDENCE_SPEC},
    {.sku="235u", .brand_prefix="ultra5",
     .p=2, .atom=10, .take_p=2, .take_atom=4, .shared_l3_atom_count=8,
     .evidence=XE_CPU_EVIDENCE_SPEC},
    {.sku="268v", .brand_prefix="ultra7",
     .p=4, .atom=4, .take_p=4,
     .evidence=XE_CPU_EVIDENCE_SPEC},
    {.sku="315", .brand_prefix="core5",
     .p=2, .atom=4, .take_p=2,
     .evidence=XE_CPU_EVIDENCE_SPEC},
    {.sku="358h", .brand_prefix="ultrax7",
     .p=4, .atom=12, .take_p=4, .take_atom=8, .shared_l3_atom_count=8,
     .evidence=XE_CPU_EVIDENCE_MEASURED_PROFILE},
};

/* Ignore trademark decorations and the optional word Processor; otherwise
 * preserve token boundaries, preventing e.g. i7-13700 matching i7-13700H. */
static int brand_tokens(const char *s, char tokens[][32], int cap) {
    int n = 0;
    if (!s) return 0;
    while (*s) {
        if (*s == '@') break;
        if (*s == '(' && (s[1] == 'R' || s[1] == 'r') && s[2] == ')') { s += 3; continue; }
        if (*s == '(' && (s[1] == 'T' || s[1] == 't') &&
            (s[2] == 'M' || s[2] == 'm') && s[3] == ')') { s += 4; continue; }
        if (!isalnum((unsigned char)*s)) { s++; continue; }
        char token[32]; size_t len = 0;
        while (isalnum((unsigned char)*s) || *s == '-') {
            if (len + 1 >= sizeof token) return 0;
            token[len++] = (char)tolower((unsigned char)*s++);
        }
        token[len] = 0;
        if (!strcmp(token, "processor") || !strcmp(token, "cpu")) continue;
        if (n == cap) return 0;
        memcpy(tokens[n++], token, len + 1);
    }
    return n;
}

static int policy_brand_matches(const policy *p, char t[][32], int n) {
    /* Linux exposes e.g. "13th Gen Intel(R) Core(TM) i7-13700H". */
    if (n >= 4 && !strcmp(t[1], "gen")) {
        const char *q = t[0];
        while (isdigit((unsigned char)*q)) q++;
        if (q == t[0] || (strcmp(q, "th") && strcmp(q, "st") &&
                          strcmp(q, "nd") && strcmp(q, "rd"))) return 0;
        t += 2;
        n -= 2;
    }
    if (n < 2 || strcmp(t[0], "intel")) return 0;
    if (!strcmp(p->brand_prefix, "n"))
        return n == 2 && !strcmp(t[1], p->sku);
    if (!strcmp(p->brand_prefix, "core"))
        return n == 3 && !strcmp(t[1], "core") && !strcmp(t[2], p->sku);
    if (!strcmp(p->brand_prefix, "core5"))
        return n == 4 && !strcmp(t[1], "core") && !strcmp(t[2], "5") && !strcmp(t[3], p->sku);
    const char *tier = !strcmp(p->brand_prefix, "ultra5") ? "5" :
                       !strcmp(p->brand_prefix, "ultra7") ? "7" : "x7";
    return n == 5 && !strcmp(t[1], "core") && !strcmp(t[2], "ultra") &&
           !strcmp(t[3], tier) && !strcmp(t[4], p->sku);
}

static xe_cpu_policy_result fail(xe_cpu_policy_status status) {
    return (xe_cpu_policy_result){status, XE_CPU_EVIDENCE_NONE, NULL, 0};
}

/* No supported profile selects more than this many cores. */
enum { XE_CPU_POLICY_MAX_CHOSEN = 12 };
xe_cpu_policy_result xe_cpu_policy_select(const xe_cpu_inventory *inv,
                                          int *out, size_t capacity) {
    if (!inv || !inv->intel_vendor || !inv->identity_verified)
        return fail(XE_CPU_POLICY_UNVERIFIED);
    char tokens[16][32];
    int nt = brand_tokens(inv->brand, tokens, 16);
    const policy *p = NULL;
    for (size_t i = 0; i < sizeof policies / sizeof policies[0]; i++) {
        if (policy_brand_matches(&policies[i], tokens, nt)) {
            if (p) return fail(XE_CPU_POLICY_UNKNOWN_SKU);
            p = &policies[i];
        }
    }
    if (!p) return fail(XE_CPU_POLICY_UNKNOWN_SKU);
    if (!inv->complete) return fail(XE_CPU_POLICY_INCOMPLETE);
    if (!inv->cores || !inv->core_count) return fail(XE_CPU_POLICY_INVALID_INVENTORY);
    int nc = 0, np = 0, na = 0;
    for (size_t i = 0; i < inv->core_count; i++) {
        const xe_cpu_core *c = &inv->cores[i];
        if (c->cpu < 0 || c->package < 0 || c->core < 0 ||
            c->class_id < XE_CPU_HOMOGENEOUS || c->class_id > XE_CPU_ATOM ||
            c->l2_group < XE_CPU_CACHE_ABSENT || c->l3_group < XE_CPU_CACHE_ABSENT ||
            c->package != inv->cores[0].package)
            return fail(XE_CPU_POLICY_INVALID_INVENTORY);
        for (size_t j = 0; j < i; j++)
            if (c->cpu == inv->cores[j].cpu ||
                (c->package == inv->cores[j].package && c->core == inv->cores[j].core))
                return fail(XE_CPU_POLICY_INVALID_INVENTORY);
        nc += c->class_id == XE_CPU_HOMOGENEOUS;
        np += c->class_id == XE_CPU_P;
        na += c->class_id == XE_CPU_ATOM;
    }
    int homogeneous_atom = p->use_homogeneous_atom && nc == p->atom && np == 0 && na == 0;
    /* Some P-only profiles may not expose a hybrid class. */
    int homogeneous_p = p->use_homogeneous_p && nc == p->p && np == 0 && na == 0;
    if (!homogeneous_atom && !homogeneous_p &&
        (nc != p->homogeneous || np != p->p || na != p->atom))
        return fail(XE_CPU_POLICY_TOPOLOGY_MISMATCH);
    int common_l3 = XE_CPU_CACHE_UNKNOWN;
    if (p->shared_l3_atom_count) {
        for (size_t i = 0; i < inv->core_count; i++) {
            const xe_cpu_core *c = &inv->cores[i];
            if (c->l3_group == XE_CPU_CACHE_UNKNOWN) return fail(XE_CPU_POLICY_CACHE_AMBIGUOUS);
            if (c->class_id == XE_CPU_P) {
                if (c->l3_group < 0) return fail(XE_CPU_POLICY_CACHE_AMBIGUOUS);
                if (common_l3 == XE_CPU_CACHE_UNKNOWN) common_l3 = c->l3_group;
                else if (common_l3 != c->l3_group) return fail(XE_CPU_POLICY_CACHE_AMBIGUOUS);
            }
        }
        int shared = 0;
        for (size_t i = 0; i < inv->core_count; i++)
            if (inv->cores[i].class_id == XE_CPU_ATOM && inv->cores[i].l3_group == common_l3) shared++;
        if (shared != p->shared_l3_atom_count) return fail(XE_CPU_POLICY_CACHE_AMBIGUOUS);
    }
    size_t total = (size_t)(p->take_homogeneous + p->take_p + p->take_atom);
    if (!out || capacity < total) return fail(XE_CPU_POLICY_OUTPUT_TOO_SMALL);
    int chosen[XE_CPU_POLICY_MAX_CHOSEN]; size_t used = 0;
    /* A complete L2 group is identified by the collector. Prefer a group
     * with enough eligible Atom cores; ties use the smallest group ID. */
    int preferred_l2 = XE_CPU_CACHE_UNKNOWN;
    if (p->take_atom == 4 && !homogeneous_atom) {
        for (size_t i = 0; i < inv->core_count; i++) {
            const xe_cpu_core *c = &inv->cores[i];
            if (c->class_id != XE_CPU_ATOM || c->l2_group < 0 ||
                (p->shared_l3_atom_count && c->l3_group != common_l3)) continue;
            int group_count = 0;
            for (size_t j = 0; j < inv->core_count; j++) {
                const xe_cpu_core *d = &inv->cores[j];
                if (d->class_id == XE_CPU_ATOM && d->l2_group == c->l2_group &&
                    (!p->shared_l3_atom_count || d->l3_group == common_l3)) group_count++;
            }
            if (group_count >= 4 && (preferred_l2 < 0 || c->l2_group < preferred_l2))
                preferred_l2 = c->l2_group;
        }
    }
    for (int phase = 0; phase < 3; phase++) {
        xe_cpu_class cls = phase == 0 ? XE_CPU_HOMOGENEOUS : phase == 1 ? XE_CPU_P : XE_CPU_ATOM;
        int take = phase == 0 ? p->take_homogeneous : phase == 1 ? p->take_p : p->take_atom;
        if (homogeneous_atom && phase == 0) take = p->take_atom;
        if (homogeneous_atom && phase == 2) take = 0;
        if (homogeneous_p && phase == 0) take = p->take_p;
        if (homogeneous_p && phase == 1) take = 0;
        int previous = -1;
        for (int k = 0; k < take; k++) {
            int best = -1;
            for (size_t i = 0; i < inv->core_count; i++) {
                const xe_cpu_core *c = &inv->cores[i];
                if (c->class_id != cls || c->cpu <= previous ||
                    (phase == 2 && p->shared_l3_atom_count && c->l3_group != common_l3) ||
                    (phase == 2 && preferred_l2 >= 0 && c->l2_group != preferred_l2)) continue;
                if (best < 0 || c->cpu < best) best = c->cpu;
            }
            if (best < 0) return fail(XE_CPU_POLICY_TOPOLOGY_MISMATCH);
            chosen[used++] = best; previous = best;
        }
    }
    memcpy(out, chosen, used * sizeof *out);
    return (xe_cpu_policy_result){XE_CPU_POLICY_MATCH, p->evidence, p->sku, used};
}

const char *xe_cpu_policy_reason(xe_cpu_policy_status status) {
    switch (status) {
    case XE_CPU_POLICY_MATCH: return "catalog match";
    case XE_CPU_POLICY_UNVERIFIED: return "CPU identity unverified";
    case XE_CPU_POLICY_UNKNOWN_SKU: return "CPU SKU not in catalog";
    case XE_CPU_POLICY_INCOMPLETE: return "incomplete CPU inventory";
    case XE_CPU_POLICY_INVALID_INVENTORY: return "invalid CPU inventory";
    case XE_CPU_POLICY_TOPOLOGY_MISMATCH: return "CPU topology does not match SKU";
    case XE_CPU_POLICY_CACHE_AMBIGUOUS: return "shared cache topology ambiguous";
    case XE_CPU_POLICY_OUTPUT_TOO_SMALL: return "selection buffer too small";
    }
    return "unknown policy status";
}
