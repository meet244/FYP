/** @type {import('next').NextConfig} */
const nextConfig = {
  // The v0 scaffold this was lifted from set ignoreBuildErrors:true. Left off
  // here — the API client is a hand-maintained mirror of the backend's Pydantic
  // schemas, and a type error in it is exactly the bug worth failing the build.
  images: {
    unoptimized: true,
  },
}

export default nextConfig
