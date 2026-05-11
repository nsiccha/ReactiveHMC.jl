using Documenter, DocumenterVitepress, ReactiveHMC
import HTMXObjects

# Sync HTMXObjects' canonical `htmxo-embed.ts` (+ companion CSS) into our
# theme dir before DocumenterVitepress runs. The theme's `index.ts`
# imports `setupHtmxoEmbed` from it; calling this in make.jl keeps the
# wiring auto-updated when HTMXObjects ships a new embed runtime.
HTMXObjects.vitepress_theme_install(joinpath(@__DIR__, "src", ".vitepress", "theme"))

makedocs(
    sitename = "ReactiveHMC.jl",
    modules  = [ReactiveHMC],
    format   = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/nsiccha/ReactiveHMC.jl",
        devurl = "dev",
        devbranch = "dev",
    ),
    pages = [
        "Home"      => "index.md",
        "Gallery"   => "gallery.md",
        "API"       => "api.md",
    ],
    checkdocs = :none,
    warnonly = true,
)

let redirect = joinpath(@__DIR__, "build", "index.html")
    isfile(redirect) || write(redirect, """
    <!DOCTYPE html>
    <html><head>
    <meta http-equiv="refresh" content="0; url=dev/">
    </head><body>Redirecting to <a href="dev/">dev</a>...</body></html>
    """)
end

DocumenterVitepress.deploydocs(
    repo = "github.com/nsiccha/ReactiveHMC.jl",
    devbranch = "dev",
    push_preview = true,
)
