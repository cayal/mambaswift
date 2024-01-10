// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Mamba",
    platforms: [SupportedPlatform.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.2.0"),
        .package(url: "https://github.com/cayal/BPETokenizer", revision: "d4cd38e51bcbe8b51a54da4bdf6d8d6d9043aec7"),
        .package(url: "https://github.com/cayal/mambuflo", revision: "09d7f73616aa81b5028fd578c1e04bcb1186bde6"),
    ],
    targets: [
        .executableTarget(
            name: "Demo",
            dependencies: [
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "BPETokenizer", package: "BPETokenizer"),
                .product(name: "MamBufLo", package: "MamBufLo")
            ],
            exclude: ["./Extra"],
            resources: [
                .copy("./Resources/special_tokens_map.json"),
                .copy("./Resources/tokenizer.json"),
                .copy("./Resources/MmMamba.metallib"),
                .copy("./Resources/converted")
            ],
            swiftSettings: [.enableUpcomingFeature("BareSlashRegexLiterals")]
        ),
        .testTarget(name: "Tests",
                    dependencies: [.target(name: "Demo")],
                    resources: [
                        .copy("./Resources/special_tokens_map.json"),
                        .copy("./Resources/tokenizer.json"),
                        .copy("./Resources/MmMamba.metallib")
                    ])
    ]
)
