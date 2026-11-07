import re
from collections.abc import Callable
from dataclasses import asdict, dataclass

MENTION_RE = re.compile(r"@([\w.-]+)")


@dataclass(slots=True)
class Post:
    id: int
    author_id: int
    thread_id: int
    parent_id: int | None
    title: str | None
    body: str
    created_at: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class BoardError(ValueError):
    pass


class Board:
    """Public message board: top-level posts are threads; replies hang off a thread."""

    def __init__(self, max_title: int = 120, max_body: int = 600) -> None:
        self.max_title = max_title
        self.max_body = max_body
        self._posts: dict[int, Post] = {}
        self._replies: dict[int, list[int]] = {}
        self._next_id = 1

    def _new_post(
        self,
        author_id: int,
        thread_id: int | None,
        parent_id: int | None,
        title: str | None,
        body: str,
        now: float,
    ) -> Post:
        body = body.strip()
        if not body:
            raise BoardError("body must not be empty")
        if len(body) > self.max_body:
            raise BoardError(f"body longer than {self.max_body} characters")
        pid = self._next_id
        self._next_id += 1
        post = Post(pid, author_id, thread_id or pid, parent_id, title, body, now)
        self._posts[pid] = post
        return post

    def clear(self) -> None:
        """Forget every post but keep issuing fresh ids, so old links never point at new threads."""
        self._posts.clear()
        self._replies.clear()

    def post_thread(self, author_id: int, title: str, body: str, now: float) -> Post:
        title = title.strip()
        if not title:
            raise BoardError("title must not be empty")
        if len(title) > self.max_title:
            raise BoardError(f"title longer than {self.max_title} characters")
        post = self._new_post(author_id, None, None, title, body, now)
        self._replies[post.id] = []
        return post

    def reply(
        self, author_id: int, thread_id: int, body: str, now: float, parent_id: int | None = None
    ) -> Post:
        if thread_id not in self._replies:
            raise BoardError(f"thread {thread_id} does not exist")
        if parent_id is not None:
            parent = self._posts.get(parent_id)
            if parent is None or parent.thread_id != thread_id:
                raise BoardError(f"post {parent_id} is not in thread {thread_id}")
        post = self._new_post(author_id, thread_id, parent_id, None, body, now)
        self._replies[thread_id].append(post.id)
        return post

    def get(self, post_id: int) -> Post | None:
        return self._posts.get(post_id)

    def thread(self, thread_id: int) -> list[Post]:
        """Root post followed by its replies in order; empty if the thread is unknown."""
        if thread_id not in self._replies:
            return []
        return [self._posts[thread_id]] + [self._posts[i] for i in self._replies[thread_id]]

    def threads(self) -> list[Post]:
        return [self._posts[t] for t in self._replies]

    def reply_count(self, thread_id: int) -> int:
        return len(self._replies.get(thread_id, ()))

    def last_activity(self, thread_id: int) -> float:
        replies = self._replies[thread_id]
        last = replies[-1] if replies else thread_id
        return self._posts[last].created_at

    def digest(
        self, limit: int = 10, name_of: Callable[[int], str] = lambda i: f"agent {i}"
    ) -> list[dict[str, object]]:
        """The most recently active threads, summarised for an agent's observation."""
        ordered = sorted(self._replies, key=self.last_activity, reverse=True)[:limit]
        out: list[dict[str, object]] = []
        for tid in ordered:
            root = self._posts[tid]
            replies = self._replies[tid]
            latest = self._posts[replies[-1]] if replies else None
            out.append(
                {
                    "thread_id": tid,
                    "title": root.title,
                    "author": name_of(root.author_id),
                    "replies": len(replies),
                    "latest": (
                        {"author": name_of(latest.author_id), "body": latest.body}
                        if latest
                        else None
                    ),
                }
            )
        return out

    def mentions_in(self, text: str) -> list[str]:
        return MENTION_RE.findall(text)
