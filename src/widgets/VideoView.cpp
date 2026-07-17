#include "VideoView.h"

#include <QResizeEvent>

VideoView::VideoView(QWidget *parent)
    : QLabel(parent)
{
    setAlignment(Qt::AlignCenter);
    setMinimumSize(320, 240);
    setStyleSheet(QStringLiteral("background-color: #1a1a1a; color: #888888;"));
    setText(QStringLiteral("No video source"));
}

void VideoView::setFrame(const QImage &image)
{
    m_lastFrame = image;
    updatePixmap();
}

void VideoView::resizeEvent(QResizeEvent *event)
{
    QLabel::resizeEvent(event);
    updatePixmap();
}

void VideoView::updatePixmap()
{
    if (m_lastFrame.isNull())
        return;
    setPixmap(QPixmap::fromImage(m_lastFrame).scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}
